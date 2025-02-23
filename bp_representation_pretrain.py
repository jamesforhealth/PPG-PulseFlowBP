import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

##############################################
# 全域參數：ABP 反還原參數（請根據您實際縮放方式調整）
##############################################
ABP_OFFSET = 50.0
ABP_SCALE = 100.0

##############################################
# 0. DataSet 定義：讀取 h5 檔資料，並對 abp 與 bp_label 進行正規化
##############################################
class VitalSignDataset(Dataset):
    """
    從 h5 檔讀取：
      - 'ppg': (N,1250)
      - 'ecg': (N,1250) 若不存在則補 0
      - 'abp': (N,1250)（原始 ABP 波型，以 mmHg 存放，讀入後正規化到 [0,1]）
      - 若檔案中有 'segsbp' 與 'segdbp'，則作為對比學習標籤，
        否則利用 abp 平均值複製兩次。
    """
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            self.ppg = torch.from_numpy(f['ppg'][:]).float()    # (N,1250)
            if 'ecg' in f:
                self.ecg = torch.from_numpy(f['ecg'][:]).float()  # (N,1250)
            else:
                self.ecg = torch.zeros_like(self.ppg)
            self.abp = torch.from_numpy(f['abp'][:]).float()      # (N,1250)
            # 正規化：將原始 ABP 轉換到 [0,1]
            self.abp = (self.abp - ABP_OFFSET) / ABP_SCALE
            if 'segsbp' in f and 'segdbp' in f:
                self.segsbp = torch.from_numpy(f['segsbp'][:]).float()  # (N,)
                self.segdbp = torch.from_numpy(f['segdbp'][:]).float()  # (N,)
                bp_label = torch.stack([self.segsbp, self.segdbp], dim=1)  # (N,2)
            else:
                mean_abp = self.abp.mean(dim=-1, keepdim=True)
                bp_label = torch.cat([mean_abp, mean_abp], dim=1)  # (N,2)
            # 正規化 bp_label
            self.bp_label = (bp_label - ABP_OFFSET) / ABP_SCALE
        self.N = self.ppg.shape[0]
        # 加上 channel 維度
        self.ppg = self.ppg.unsqueeze(1)  # (N,1,1250)
        self.ecg = self.ecg.unsqueeze(1)  # (N,1,1250)
        # 合併成 (N,2,1250)
        self.signal = torch.cat([self.ppg, self.ecg], dim=1)
        self.abp = self.abp.unsqueeze(1)  # (N,1,1250)
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return {
            'signal': self.signal[idx],      # (2,1250)
            'abp': self.abp[idx],            # (1,1250)
            'bp_label': self.bp_label[idx]   # (2,)
        }

##############################################
# 1. 模型架構：利用 ResUNet+TCN 建立 Encoder，Projection Head 做對比學習，Decoder 重建 ABP
##############################################

# 1.1 基本 1D 卷積區塊
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 1.2 DownBlock 與 UpBlock
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = ConvBlock1D(in_ch, out_ch, 3)
        self.conv2 = ConvBlock1D(out_ch, out_ch, 3)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv1 = ConvBlock1D(out_ch*2, out_ch, 3)
        self.conv2 = ConvBlock1D(out_ch, out_ch, 3)
    def forward(self, x, skip):
        x = self.upconv(x)
        diff = skip.shape[-1] - x.shape[-1]
        if diff > 0:
            skip = skip[..., :x.shape[-1]]
        elif diff < 0:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# 1.3 TCN 模塊（簡單版本）
class SimpleTCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.1):
        super(SimpleTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=(kernel_size-1)*dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# 1.4 Encoder – 結合 ResUNet 與 TCN
class ResUNetTCNEncoder(nn.Module):
    def __init__(self, in_ch=2, out_ch=64, base_ch=16, tcn_channels=[64,64], latent_dim=64):
        """
        先用 ResUNet 提取特徵，再用 TCN 捕捉時序資訊，
        最後經 global average pooling 與全連接層得到 latent 向量。
        """
        super(ResUNetTCNEncoder, self).__init__()
        self.enc_conv1 = nn.Sequential(
            ConvBlock1D(in_ch, base_ch, 3),
            ConvBlock1D(base_ch, base_ch, 3)
        )
        self.down1 = DownBlock(base_ch, base_ch*2)
        self.down2 = DownBlock(base_ch*2, base_ch*4)
        self.down3 = DownBlock(base_ch*4, base_ch*8)
        self.bottleneck = nn.Sequential(
            ConvBlock1D(base_ch*8, base_ch*8, 3),
            ConvBlock1D(base_ch*8, base_ch*8, 3)
        )
        self.up1 = UpBlock(base_ch*8, base_ch*4)
        self.up2 = UpBlock(base_ch*4, base_ch*2)
        self.up3 = UpBlock(base_ch*2, base_ch)
        self.final = nn.Conv1d(base_ch, out_ch, kernel_size=1, bias=False)
        # TCN 模塊
        self.tcn = SimpleTCN(input_dim=out_ch, num_channels=tcn_channels, kernel_size=3, dropout=0.1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Linear(tcn_channels[-1], latent_dim)
    def forward(self, x):
        # x: (B,2,1250)
        c1 = self.enc_conv1(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        b = self.bottleneck(c4)
        d1 = self.up1(b, c3)
        d2 = self.up2(d1, c2)
        d3 = self.up3(d2, c1)
        out = self.final(d3)  # (B, out_ch, L)
        tcn_out = self.tcn(out)  # (B, tcn_channels[-1], L)
        pooled = self.global_pool(tcn_out).squeeze(-1)  # (B, tcn_channels[-1])
        latent = self.fc_latent(pooled)  # (B, latent_dim)
        return latent

# 1.5 Projection Head（用於對比學習）
class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=64, proj_dim=32):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )
    def forward(self, z):
        return self.mlp(z)

# 1.6 Decoder – 將 latent 向量映射回 ABP 波型 (B,1,1250)
class DecoderModule(nn.Module):
    def __init__(self, latent_dim=64, output_length=1250):
        super(DecoderModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_length)
        )
        # 使用 Sigmoid 保證輸出在 [0,1]
        self.sigmoid = nn.Sigmoid()
    def forward(self, z):
        out = self.fc(z)  # (B, output_length)
        out = self.sigmoid(out)
        out = out.unsqueeze(1)  # (B,1,output_length)
        return out

# 1.7 整體模型：ABPContrastiveAutoencoder
class ABPContrastiveAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, proj_dim=32, output_length=1250):
        super(ABPContrastiveAutoencoder, self).__init__()
        self.encoder = ResUNetTCNEncoder(in_ch=2, out_ch=64, base_ch=16,
                                         tcn_channels=[64,64],
                                         latent_dim=latent_dim)
        self.projection_head = ProjectionHead(latent_dim=latent_dim, proj_dim=proj_dim)
        self.decoder = DecoderModule(latent_dim=latent_dim, output_length=output_length)
    def forward(self, x):
        # x: (B,2,1250)
        latent = self.encoder(x)           # (B, latent_dim)
        proj = self.projection_head(latent)  # (B, proj_dim)
        recon = self.decoder(latent)         # (B,1,1250)
        return recon, latent, proj

##############################################
# 2. 損失函數
##############################################
def reconstruction_loss(recon, target):
    return F.mse_loss(recon, target)

def triplet_loss_bp(representations, bp_labels, margin=0.2, threshold=0.1):
    """
    利用 batch 中每個樣本的 representation 與對應 bp_labels (shape: (B,2))
    為每個 anchor 找出正樣本（與其 bp (SBP,DBP) 歐式距離小於 threshold）與負樣本（距離大於等於 threshold）
    """
    loss = 0.0
    valid_count = 0
    batch_size = representations.size(0)
    for i in range(batch_size):
        anchor = representations[i]
        label_anchor = bp_labels[i]  # (2,)
        dists = torch.norm(bp_labels - label_anchor, p=2, dim=1)  # (B,)
        pos_mask = (dists < threshold) & (torch.arange(batch_size, device=bp_labels.device) != i)
        neg_mask = dists >= threshold
        pos_indices = pos_mask.nonzero(as_tuple=False).squeeze()
        neg_indices = neg_mask.nonzero(as_tuple=False).squeeze()
        if pos_indices.dim() == 0:
            pos_indices = pos_indices.unsqueeze(0)
        if neg_indices.dim() == 0:
            neg_indices = neg_indices.unsqueeze(0)
        if pos_indices.numel() == 0 or neg_indices.numel() == 0:
            continue
        pos = representations[pos_indices[0]]
        neg = representations[neg_indices[0]]
        d_pos = F.pairwise_distance(anchor.unsqueeze(0), pos.unsqueeze(0))
        d_neg = F.pairwise_distance(anchor.unsqueeze(0), neg.unsqueeze(0))
        loss += F.relu(d_pos - d_neg + margin)
        valid_count += 1
    if valid_count > 0:
        return loss / valid_count
    else:
        return torch.tensor(0.0, device=representations.device)

##############################################
# 3. 回歸模型：利用 encoder 輸出的 latent 向量預測 (segsbp,segdbp)
##############################################
class BPRegressor(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=32, output_dim=2):
        super(BPRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

def train_regressor_epoch(autoencoder, regressor, train_loader, optimizer, device):
    # 在回歸訓練階段，固定 autoencoder.encoder
    autoencoder.eval()
    regressor.train()
    total_loss = 0.0
    for batch in train_loader:
        x = batch['signal'].to(device)
        labels = batch['bp_label'].to(device)
        with torch.no_grad():
            latent = autoencoder.encoder(x)
        pred = regressor(latent)
        loss = F.mse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_regressor(autoencoder, regressor, loader, device):
    autoencoder.eval()
    regressor.eval()
    mae_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch['signal'].to(device)
            labels = batch['bp_label'].to(device)
            latent = autoencoder.encoder(x)
            pred = regressor(latent)
            # 反正規化到 mmHg
            pred_mmHg = pred * ABP_SCALE + ABP_OFFSET
            labels_mmHg = labels * ABP_SCALE + ABP_OFFSET
            mae = F.l1_loss(pred_mmHg, labels_mmHg, reduction='mean')
            mae_list.append(mae.item())
    return np.mean(mae_list)

##############################################
# 4. 訓練與視覺化函式
##############################################
def visualize_and_save_validation_sample(model, val_dataloader, device='cuda', epoch=0):
    """
    從驗證集中抽一筆資料，畫出：
      1. 輸入 PPG 信號 (channel 0)
      2. 輸入 ECG 信號 (channel 1)
      3. Ground Truth ABP（反還原至 mmHg）
      4. Predicted ABP（反還原至 mmHg）
      5. Overlay：Predicted 與 Ground Truth 疊加
    將圖表存成圖片，檔名中含有 epoch 編號，並不阻塞主訓練續。
    """
    model.eval()
    for batch in val_dataloader:
        sample = {key: val[0:1] for key, val in batch.items()}
        break
    x = sample['signal'].to(device)     # (1,2,1250)
    target = sample['abp'].to(device)     # (1,1,1250)
    with torch.no_grad():
        recon, _, _ = model(x)
    # 反還原 ABP
    target_orig = target.cpu().numpy()[0,0] * ABP_SCALE + ABP_OFFSET
    recon_orig = recon.cpu().numpy()[0,0] * ABP_SCALE + ABP_OFFSET
    signal_np = x.cpu().numpy()[0]  # (2,1250)
    time_axis = np.arange(signal_np.shape[1])
    
    fig, axs = plt.subplots(3,2, figsize=(14,8))
    axs = axs.flatten()
    # 子圖1: Input PPG
    axs[0].plot(time_axis, signal_np[0], color='blue')
    axs[0].set_title("Input PPG Signal")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    # 子圖2: Input ECG
    axs[1].plot(time_axis, signal_np[1], color='green')
    axs[1].set_title("Input ECG Signal")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")
    # 子圖3: Ground Truth ABP (還原)
    axs[2].plot(time_axis, target_orig, color='black')
    axs[2].set_title("Ground Truth ABP")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("ABP (mmHg)")
    # 子圖4: Predicted ABP (還原)
    axs[3].plot(time_axis, recon_orig, color='red')
    axs[3].set_title("Predicted ABP")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("ABP (mmHg)")
    # 子圖5: Overlay: Predicted vs Ground Truth
    axs[4].plot(time_axis, target_orig, label='Ground Truth', color='black', linewidth=2)
    axs[4].plot(time_axis, recon_orig, label='Predicted', color='red', linestyle='--', linewidth=2)
    axs[4].set_title("Overlay: Predicted vs Ground Truth ABP")
    axs[4].set_xlabel("Time")
    axs[4].set_ylabel("ABP (mmHg)")
    axs[4].legend()
    # 隱藏第6個子圖
    axs[5].axis('off')
    
    plt.tight_layout()
    # 儲存圖檔到當前目錄，可根據需要修改路徑
    save_path = f"validation_epoch_{epoch}.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close(fig)

def train_contrastive_autoencoder(model, dataloader, val_dataloader=None, epochs=50, lr=1e-3, 
                                  contrastive_weight=0.1, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 建立回歸模型與其 optimizer
    bp_regressor = BPRegressor(latent_dim=64, hidden_dim=32, output_dim=2).to(device)
    bp_regressor_optimizer = optim.Adam(bp_regressor.parameters(), lr=lr)
    
    # 初始驗證損失計算（基於 normalized loss）
    if val_dataloader is not None:
        model.eval()
        init_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch['signal'].to(device)
                target = batch['abp'].to(device)
                labels = batch['bp_label'].to(device)
                recon, _, proj = model(x)
                loss_recon = reconstruction_loss(recon, target)
                loss_triplet = triplet_loss_bp(proj, labels, margin=0.2, threshold=0.1)
                init_val_loss += (loss_recon + contrastive_weight * loss_triplet).item()
        init_val_loss /= len(val_dataloader)
        print(f"Initial Validation Loss (normalized): {init_val_loss:.4f}")
    
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_triplet = 0.0
        for batch in tqdm(dataloader, desc=f"Train Epoch {epoch}"):
            x = batch['signal'].to(device)     # (B,2,1250)
            target = batch['abp'].to(device)     # (B,1,1250)
            labels = batch['bp_label'].to(device)  # (B,2)
            optimizer.zero_grad()
            recon, latent, proj = model(x)
            loss_recon = reconstruction_loss(recon, target)
            loss_triplet = triplet_loss_bp(proj, labels, margin=0.2, threshold=0.1)
            loss = loss_recon + contrastive_weight * loss_triplet
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_triplet += loss_triplet.item()
        avg_loss = total_loss/len(dataloader)
        avg_recon = total_recon/len(dataloader)
        avg_triplet = total_triplet/len(dataloader)
        print(f"Epoch {epoch}/{epochs} Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Triplet: {avg_triplet:.4f})")
        
        if val_dataloader is not None:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    x = batch['signal'].to(device)
                    target = batch['abp'].to(device)
                    labels = batch['bp_label'].to(device)
                    recon, latent, proj = model(x)
                    loss_recon = reconstruction_loss(recon, target)
                    loss_triplet = triplet_loss_bp(proj, labels, margin=0.2, threshold=0.1)
                    loss = loss_recon + contrastive_weight * loss_triplet
                    val_loss_sum += loss.item()
            avg_val_loss = val_loss_sum/len(val_dataloader)
            print(f"Epoch {epoch}/{epochs} Val Loss (normalized): {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = model.state_dict()
                torch.save(best_state, f"best_model_epoch_{epoch}.pth")
        
        # 執行一個 epoch 的回歸模型訓練，利用 encoder 輸出的 latent 向量預測 bp_label
        reg_train_loss = train_regressor_epoch(model, bp_regressor, dataloader, bp_regressor_optimizer, device)
        val_mae = evaluate_regressor(model, bp_regressor, val_dataloader, device) if val_dataloader is not None else None
        print(f"Epoch {epoch}/{epochs} Regression - Train Loss: {reg_train_loss:.4f}, Val MAE (mmHg): {val_mae:.2f}" if val_mae is not None else f"Epoch {epoch}/{epochs} Regression - Train Loss: {reg_train_loss:.4f}")
        
        # 每個 epoch 結束後從驗證集中抽一筆資料進行視覺化並儲存圖片
        visualize_and_save_validation_sample(model, val_dataloader, device=device, epoch=epoch)
    if val_dataloader is not None and best_val_loss < float('inf'):
        model.load_state_dict(best_state)
    return model

##############################################
# 5. DataLoader 建立
##############################################
def create_dataloaders():
    data_dir = Path('training_data_VitalDB_quality')
    train_files = [data_dir / f"training_{i+1}.h5" for i in range(9)]
    val_file = data_dir / 'validation.h5'
    test_file = data_dir / 'test.h5'
    
    train_ds_list = []
    for tf in train_files:
        if tf.exists():
            train_ds_list.append(VitalSignDataset(str(tf)))
        else:
            print(f"Warning: Training file {tf} not found")
    if not train_ds_list:
        raise FileNotFoundError("No training files found.")
    train_dataset = ConcatDataset(train_ds_list)
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found at {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found at {test_file}")
    val_dataset = VitalSignDataset(str(val_file))
    test_dataset = VitalSignDataset(str(test_file))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader, test_loader

##############################################
# 6. 主程式
##############################################
def main():
    train_loader, val_loader, test_loader = create_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 建立模型：ABPContrastiveAutoencoder
    model = ABPContrastiveAutoencoder(latent_dim=64, proj_dim=32, output_length=1250)
    # sum of parameters
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("Start training contrastive autoencoder for ABP reconstruction...")
    model = train_contrastive_autoencoder(model, train_loader, val_dataloader=val_loader, 
                                          epochs=50, lr=1e-3, contrastive_weight=0.1, device=device)
    
    # 驗證效果：視覺化部分預測結果（已在訓練過程中儲存圖片）
    visualize_and_save_validation_sample(model, val_loader, device=device)
    
    # 在測試集上計算重建 MSE（normalized loss）
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch['signal'].to(device)
            target = batch['abp'].to(device)
            recon, _, _ = model(x)
            loss = reconstruction_loss(recon, target)
            total_loss += loss.item()
    avg_test_loss = total_loss / len(test_loader)
    print(f"Test MSE (normalized): {avg_test_loss:.4f}")
    
    # 另外，計算還原到 mmHg 的 MAE 與 std
    mae_list = []
    all_errors = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['signal'].to(device)
            target = batch['abp'].to(device)
            recon, _, _ = model(x)
            target_orig = target * ABP_SCALE + ABP_OFFSET
            recon_orig = recon * ABP_SCALE + ABP_OFFSET
            error = torch.abs(recon_orig - target_orig)
            mae_list.append(error.mean().item())
            all_errors.extend(error.view(-1).cpu().numpy().tolist())
    overall_mae = np.mean(mae_list)
    overall_std = np.std(all_errors)
    print(f"Test MAE (mmHg): {overall_mae:.2f}, STD: {overall_std:.2f}")

if __name__ == "__main__":
    main()
