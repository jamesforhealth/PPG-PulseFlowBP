import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

############################################################
# 1. 自監督表徵學習模型
############################################################

class PPGEncoder(nn.Module):
    """PPG信號的編碼器"""
    def __init__(self, in_ch=1, base_ch=16, out_ch=128):
        super().__init__()
        # 第一層捕獲基本特徵
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True)
        )
        # 逐步減小時間維度，增加通道數
        self.layer1 = self._make_layer(base_ch, base_ch, stride=2)
        self.layer2 = self._make_layer(base_ch, base_ch*2, stride=2)
        self.layer3 = self._make_layer(base_ch*2, base_ch*4, stride=2)
        self.layer4 = self._make_layer(base_ch*4, base_ch*8, stride=2)
        
        # 全局特徵
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_fc = nn.Linear(base_ch*8, out_ch)
        
    def _make_layer(self, in_ch, out_ch, stride):
        layer = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        return layer
    
    def forward(self, x):
        # 特徵提取
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局特徵
        feature_map = x
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.final_fc(x)
        
        return x, feature_map

class PPGDecoder(nn.Module):
    """PPG信號的解碼器，用於重建任務"""
    def __init__(self, latent_dim=128, base_ch=128, out_ch=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch)
        
        # 逐步增加時間維度
        self.upconv1 = self._make_upconv(base_ch, base_ch//2)
        self.upconv2 = self._make_upconv(base_ch//2, base_ch//4)
        self.upconv3 = self._make_upconv(base_ch//4, base_ch//8)
        self.upconv4 = self._make_upconv(base_ch//8, base_ch//16)
        
        # 最終輸出層
        self.final_conv = nn.Conv1d(base_ch//16, out_ch, kernel_size=3, padding=1)
        
    def _make_upconv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z):
        x = self.fc(z).unsqueeze(-1)  # (B, base_ch, 1)
        
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        
        x = self.final_conv(x)
        return x

class MaskedAutoencoderPPG(nn.Module):
    """時序遮罩自編碼器用於PPG表徵學習"""
    def __init__(self, mask_ratio=0.75, encoder_dim=128, base_ch=16):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = PPGEncoder(in_ch=1, base_ch=base_ch, out_ch=encoder_dim)
        self.decoder = PPGDecoder(latent_dim=encoder_dim, base_ch=base_ch*8, out_ch=1)
        
    def _mask_signal(self, x, mask_ratio):
        """將PPG信號按照時間分割成塊，並隨機遮罩部分時間塊"""
        B, C, L = x.shape
        
        # 將時間維度分割成塊
        patch_size = 16  # 每個塊包含16個時間點
        num_patches = L // patch_size
        
        # 決定哪些塊被遮罩
        keep_num = int(num_patches * (1 - mask_ratio))
        noise = torch.randn(B, num_patches, device=x.device)  # 為每個batch和塊生成隨機值
        ids_shuffle = torch.argsort(noise, dim=1)  # 按照隨機值排序
        ids_keep = ids_shuffle[:, :keep_num]  # 保留排序後前keep_num個塊
        
        # 創建遮罩
        mask = torch.ones((B, L), device=x.device)
        for i in range(B):
            for idx in ids_keep[i]:
                start_idx = idx * patch_size
                end_idx = min(start_idx + patch_size, L)
                mask[i, start_idx:end_idx] = 0  # 標記為不遮罩
                
        # 應用遮罩（被遮罩的部分設置為0）
        masked_x = x.clone()
        masked_x = masked_x * (1 - mask.unsqueeze(1))
        
        return masked_x, mask
    
    def forward(self, x):
        # 應用遮罩
        masked_x, mask = self._mask_signal(x, self.mask_ratio)
        
        # 編碼
        latent, _ = self.encoder(masked_x)
        
        # 解碼並重建
        recon_x = self.decoder(latent)
        
        return recon_x, masked_x, mask, latent

############################################################
# 2. 血壓預測模型
############################################################

class BPPredictionModel(nn.Module):
    """血壓預測模型，包含預訓練的PPG編碼器"""
    def __init__(self, latent_dim=128, base_ch=16, info_dim=5, vascular_dim=3):
        super().__init__()
        # 加載預訓練編碼器
        self.ppg_encoder = PPGEncoder(in_ch=1, base_ch=base_ch, out_ch=latent_dim)
        
        # 自注意力機制增強特徵
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        
        # 處理個人信息
        self.info_fc = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 處理血管特性
        self.vascular_fc = nn.Sequential(
            nn.Linear(vascular_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 融合所有特徵並預測
        self.fusion_fc = nn.Sequential(
            nn.Linear(latent_dim + 32 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # 預測SBP和DBP
        )
    
    def forward(self, ppg, personal_info, vascular):
        # 編碼PPG信號
        ppg_features, _ = self.ppg_encoder(ppg)
        
        # 增強PPG特徵
        ppg_features_enhanced = ppg_features.unsqueeze(1)  # 添加序列維度
        ppg_features_enhanced, _ = self.attention(ppg_features_enhanced, ppg_features_enhanced, ppg_features_enhanced)
        ppg_features_enhanced = ppg_features_enhanced.squeeze(1)
        
        # 處理個人信息和血管特性
        info_features = self.info_fc(personal_info)
        vascular_features = self.vascular_fc(vascular)
        
        # 融合特徵
        combined_features = torch.cat([ppg_features_enhanced, info_features, vascular_features], dim=1)
        
        # 預測血壓
        bp_pred = self.fusion_fc(combined_features)
        
        return bp_pred

############################################################
# 3. 數據集和數據載入器
############################################################

class BPDataset(Dataset):
    """從 .h5 中讀取血壓數據集"""
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            self.ppg = torch.from_numpy(f['ppg'][:])  # (N,1250)
            if 'ecg' in f:
                self.ecg = torch.from_numpy(f['ecg'][:])
            else:
                self.ecg = torch.zeros_like(self.ppg)
            self.sbp = torch.from_numpy(f['segsbp'][:])
            self.dbp = torch.from_numpy(f['segdbp'][:])
            if 'personal_info' in f:
                self.personal_info = torch.from_numpy(f['personal_info'][:])
            else:
                n = self.ppg.shape[0]
                self.personal_info = torch.zeros((n,5))
            if 'vascular_properties' in f:
                self.vascular = torch.from_numpy(f['vascular_properties'][:])
            else:
                n = self.ppg.shape[0]
                self.vascular = torch.zeros((n,3))
        # reshape => (N,1,1250)
        self.ppg = self.ppg.unsqueeze(1)
        self.ecg = self.ecg.unsqueeze(1)
        # 組成 bp tensor shape=(N,2)
        self.bp_2d = torch.stack([self.sbp, self.dbp], dim=1)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx],                   # shape=(1,1250)
            'ecg': self.ecg[idx],                   # shape=(1,1250)
            'bp_values': self.bp_2d[idx],           # shape=(2,)
            'personal_info': self.personal_info[idx],  # shape=(M,)
            'vascular': self.vascular[idx]          # shape=(3,)
        }

def get_dataloaders(fold_path, batch_size=32):
    """創建訓練、驗證和測試數據載入器"""
    # 讀取訓練文件
    train_files = [fold_path / f"training_{i}.h5" for i in range(1, 10)]
    train_ds_list = []
    for tf in train_files:
        if tf.exists():
            train_ds_list.append(BPDataset(tf))
    
    train_dataset = ConcatDataset(train_ds_list)
    val_dataset = BPDataset(fold_path / 'validation.h5')
    test_dataset = BPDataset(fold_path / 'test.h5')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader

############################################################
# 4. 自監督預訓練
############################################################

def pretrain_MAE(train_loader, val_loader, device='cuda', epochs=50, lr=1e-3, mask_ratio=0.75):
    """預訓練MAE自編碼器"""
    model = MaskedAutoencoderPPG(mask_ratio=mask_ratio).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 記錄訓練過程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # 訓練
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            ppg = batch['ppg'].float().to(device)
            
            # 前向傳播
            recon_ppg, masked_ppg, mask, _ = model(ppg)
            
            # 只計算被遮罩部分的重建損失
            loss = nn.MSELoss()(recon_ppg * mask.unsqueeze(1), ppg * mask.unsqueeze(1))
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 驗證
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                ppg = batch['ppg'].float().to(device)
                
                # 前向傳播
                recon_ppg, masked_ppg, mask, _ = model(ppg)
                
                # 只計算被遮罩部分的重建損失
                loss = nn.MSELoss()(recon_ppg * mask.unsqueeze(1), ppg * mask.unsqueeze(1))
                
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新學習率
        scheduler.step()
        
        # 打印進度
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mae_model.pth")
            print(f"Model saved at epoch {epoch} with val loss {val_loss:.6f}")
    
    # 載入最佳模型
    model.load_state_dict(torch.load("best_mae_model.pth"))
    
    # 繪製損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MAE Pretraining Loss')
    plt.legend()
    plt.savefig('mae_pretraining_loss.png')
    plt.close()
    
    return model

############################################################
# 5. 微調血壓預測模型
############################################################

def train_bp_prediction(pretrained_encoder, train_loader, val_loader, test_loader, device='cuda', epochs=50, lr=5e-4):
    """微調血壓預測模型"""
    # 創建血壓預測模型，使用預訓練的編碼器
    model = BPPredictionModel().to(device)
    
    # 加載預訓練的編碼器權重
    if pretrained_encoder is not None:
        model.ppg_encoder.load_state_dict(pretrained_encoder.encoder.state_dict())
    
    # 優化器，較小的學習率用於預訓練部分
    encoder_params = list(model.ppg_encoder.parameters())
    rest_params = [p for p in model.parameters() if p not in set(encoder_params)]
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.1},  # 較小的學習率
        {'params': rest_params}
    ], lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 記錄訓練過程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # 訓練
        model.train()
        train_loss = 0
        train_mae = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            ppg = batch['ppg'].float().to(device)
            personal_info = batch['personal_info'].float().to(device)
            vascular = batch['vascular'].float().to(device)
            bp_values = batch['bp_values'].float().to(device)
            
            # 前向傳播
            bp_pred = model(ppg, personal_info, vascular)
            
            # 計算損失
            loss = nn.MSELoss()(bp_pred, bp_values)
            mae = torch.abs(bp_pred - bp_values).mean()
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mae.item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_losses.append(train_loss)
        
        # 驗證
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                ppg = batch['ppg'].float().to(device)
                personal_info = batch['personal_info'].float().to(device)
                vascular = batch['vascular'].float().to(device)
                bp_values = batch['bp_values'].float().to(device)
                
                # 前向傳播
                bp_pred = model(ppg, personal_info, vascular)
                
                # 計算損失
                loss = nn.MSELoss()(bp_pred, bp_values)
                mae = torch.abs(bp_pred - bp_values).mean()
                
                val_loss += loss.item()
                val_mae += mae.item()
                
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新學習率
        scheduler.step(val_loss)
        
        # 打印進度
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), "best_bp_model.pth")
            print(f"Model saved at epoch {epoch} with val loss {val_loss:.6f}")
    
    # 載入最佳模型
    model.load_state_dict(best_model_state)
    
    # 繪製損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('BP Prediction Loss')
    plt.legend()
    plt.savefig('bp_prediction_loss.png')
    plt.close()
    
    # 測試模型
    model.eval()
    test_loss = 0
    test_mae = 0
    test_sbp_mae = 0
    test_dbp_mae = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ppg = batch['ppg'].float().to(device)
            personal_info = batch['personal_info'].float().to(device)
            vascular = batch['vascular'].float().to(device)
            bp_values = batch['bp_values'].float().to(device)
            
            # 前向傳播
            bp_pred = model(ppg, personal_info, vascular)
            
            # 計算損失
            loss = nn.MSELoss()(bp_pred, bp_values)
            mae = torch.abs(bp_pred - bp_values).mean()
            sbp_mae = torch.abs(bp_pred[:, 0] - bp_values[:, 0]).mean()
            dbp_mae = torch.abs(bp_pred[:, 1] - bp_values[:, 1]).mean()
            
            test_loss += loss.item()
            test_mae += mae.item()
            test_sbp_mae += sbp_mae.item()
            test_dbp_mae += dbp_mae.item()
            
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    test_sbp_mae /= len(test_loader)
    test_dbp_mae /= len(test_loader)
    
    print(f"Test Results - Loss: {test_loss:.6f}, MAE: {test_mae:.6f}, SBP MAE: {test_sbp_mae:.6f}, DBP MAE: {test_dbp_mae:.6f}")
    
    return model, (test_loss, test_mae, test_sbp_mae, test_dbp_mae)

############################################################
# 6. 主程序
############################################################

def main():
    # 設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_path = Path('training_data_VitalDB_quality')
    batch_size = 64
    
    # 創建數據載入器
    train_loader, val_loader, test_loader = get_dataloaders(fold_path, batch_size)
    print(f"Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # 自監督預訓練
    print("\n=== Starting Self-Supervised Pretraining (MAE) ===")
    pretrained_model = pretrain_MAE(train_loader, val_loader, device=device, epochs=30, lr=1e-3, mask_ratio=0.75)
    
    # 使用預訓練模型微調血壓預測
    print("\n=== Starting BP Prediction Model Fine-tuning ===")
    bp_model_with_pretrain, metrics_with_pretrain = train_bp_prediction(
        pretrained_model, train_loader, val_loader, test_loader, 
        device=device, epochs=50, lr=5e-4
    )
    
    # 比較：從頭開始訓練血壓預測（不使用預訓練）
    print("\n=== Starting BP Prediction Model Training from Scratch ===")
    bp_model_from_scratch, metrics_from_scratch = train_bp_prediction(
        None, train_loader, val_loader, test_loader,
        device=device, epochs=50, lr=1e-3
    )
    
    # 輸出比較結果
    print("\n=== Results Comparison ===")
    print("With Pretraining:")
    print(f"  Loss: {metrics_with_pretrain[0]:.6f}, MAE: {metrics_with_pretrain[1]:.6f}")
    print(f"  SBP MAE: {metrics_with_pretrain[2]:.6f}, DBP MAE: {metrics_with_pretrain[3]:.6f}")
    
    print("From Scratch:")
    print(f"  Loss: {metrics_from_scratch[0]:.6f}, MAE: {metrics_from_scratch[1]:.6f}")
    print(f"  SBP MAE: {metrics_from_scratch[2]:.6f}, DBP MAE: {metrics_from_scratch[3]:.6f}")
    
    # 保存最終模型
    torch.save(bp_model_with_pretrain.state_dict(), "final_bp_model_with_pretrain.pth")
    torch.save(bp_model_from_scratch.state_dict(), "final_bp_model_from_scratch.pth")

if __name__ == "__main__":
    main()