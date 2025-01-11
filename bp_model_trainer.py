import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from torchviz import make_dot
######################################
# 1) 資料讀取 + Dataset 定義
######################################
class MultiH5Dataset(Dataset):
    """
    將多個 BPDataset 串接起來
    """
    def __init__(self, list_of_h5):
        self.datasets = []
        for h5_file in list_of_h5:
            ds = BPDataset(h5_file)  # 這是您原本的 BPDataset
            self.datasets.append(ds)

        self._lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self._lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 判斷 idx 屬於哪個子 dataset
        offset = 0
        for ds_idx, ds in enumerate(self.datasets):
            size = self._lengths[ds_idx]
            if idx < offset + size:
                return ds[idx - offset]
            offset += size
        raise IndexError(f"Index {idx} out of range for MultiH5Dataset")

class BPDataset(Dataset):
    """
    假設 .h5 裡包含:
      - ppg: (N, 1024)
      - ecg: (N, 1024)
      - segsbp_dbp: (N, 2)  # [segsbp, segdbp]
      - personal_info: (N, M) # M維的個人資訊, 例如 [Age, BMI, Height, Weight, Gender]
      - [可選] vpg: (N, 1024)
      - [可選] apg: (N, 1024)
    若無 vpg, apg 則動態由 ppg 計算。
    """

    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            print(f"\n[Dataset] 載入資料: {self.h5_path}")
            for k in f.keys():
                print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

            self.ppg = torch.FloatTensor(f['ppg'][:])        # (N,1024)
            self.ecg = torch.FloatTensor(f['ecg'][:])        # (N,1024)
            self.segsbp = torch.FloatTensor(f['segsbp'][:])        # (N,)
            self.segdbp = torch.FloatTensor(f['segdbp'][:])        # (N,)
            self.bp_2d = torch.stack([self.segsbp, self.segdbp], dim=1)  # (N,2)

            # 若無 personal_info 就補零向量
            if 'personal_info' in f:
                self.personal_info = torch.FloatTensor(f['personal_info'][:]) # (N,M)
            else:
                print(f"[警告] '{h5_path}' 中無 personal_info，將以零向量代替")
                N = self.ppg.shape[0]
                self.personal_info = torch.zeros((N, 4))

            # 若無 vpg、apg 就用 ppg 動態計算
            if 'vpg' in f:
                self.vpg = torch.FloatTensor(f['vpg'][:])
            else:
                print("[Info] 'vpg' not found, compute from ppg.")
                self.vpg = self.calculate_first_derivative(self.ppg)

            if 'apg' in f:
                self.apg = torch.FloatTensor(f['apg'][:])
            else:
                print("[Info] 'apg' not found, compute from ppg.")
                self.apg = self.calculate_second_derivative(self.ppg)

        # shape 調整 => (N,1,1024)
        self.ppg = self.ppg.unsqueeze(1)
        self.vpg = self.vpg.unsqueeze(1)
        self.apg = self.apg.unsqueeze(1)
        self.ecg = self.ecg.unsqueeze(1)

        print(f"  => PPG shape: {self.ppg.shape}")
        print(f"  => VPG shape: {self.vpg.shape}")
        print(f"  => APG shape: {self.apg.shape}")
        print(f"  => ECG shape: {self.ecg.shape}")
        print(f"  => BP(2D) shape: {self.bp_2d.shape}")
        print(f"  => PersonalInfo shape: {self.personal_info.shape}")

    def calculate_first_derivative(self, signal):
        """
        一階導數: vpg[n] = (signal[n+1] - signal[n-1]) / 2
        首尾做簡單複製
        shape = (N,1024)
        """
        vpg = torch.zeros_like(signal)
        vpg[:, 1:-1] = (signal[:, 2:] - signal[:, :-2]) / 2
        vpg[:, 0] = vpg[:, 1]
        vpg[:, -1] = vpg[:, -2]
        return vpg

    def calculate_second_derivative(self, signal):
        """
        二階導數: apg[n] = signal[n+1] - 2*signal[n] + signal[n-1]
        shape = (N,1024)
        """
        apg = torch.zeros_like(signal)
        apg[:, 1:-1] = signal[:, 2:] - 2*signal[:, 1:-1] + signal[:, :-2]
        apg[:, 0] = apg[:, 1]
        apg[:, -1] = apg[:, -2]
        return apg

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx],          
            'vpg': self.vpg[idx],
            'apg': self.apg[idx],
            'ecg': self.ecg[idx],
            'bp_values': self.bp_2d[idx],       
            'personal_info': self.personal_info[idx] 
        }


######################################
# 2) ResNet-like Block 與分支定義
######################################
class ResBlock1D(nn.Module):
    """
    簡易 1D ResNet Block:
      2層 Conv1d + BN + ReLU + shortcut
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                               padding=kernel_size//2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.downsample = (stride != 1 or in_ch != out_ch)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)
        return out

class DeepResBranchNet(nn.Module):
    """
    多 stage + 多 block 的 1D ResNet for 單一路徑 (e.g. PPG)。
    時間維度逐漸下採樣, 通道逐漸增加 -> 大幅提升參數量 & 表現力。
    """
    def __init__(self, in_channels=1, base_filters=64, layers_per_stage=[3,4,6,3]):
        super().__init__()
        self.in_ch = base_filters

        # conv_in
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        # 以上會將原本 1024 => (大約) 256 length, 並且通道=base_filters

        # stages: 依 layers_per_stage 配置
        # stage1 => out_ch= base_filters
        # stage2 => out_ch= base_filters*2, stride=2
        # stage3 => out_ch= base_filters*4, stride=2
        # stage4 => out_ch= base_filters*8, stride=2
        self.stage1 = self._make_stage(base_filters,   base_filters,   layers_per_stage[0], stride=1)
        self.stage2 = self._make_stage(base_filters,   base_filters*2, layers_per_stage[1], stride=2)
        self.stage3 = self._make_stage(base_filters*2, base_filters*4, layers_per_stage[2], stride=2)
        self.stage4 = self._make_stage(base_filters*4, base_filters*8, layers_per_stage[3], stride=2)

        self.out_ch = base_filters*8
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks = []
        # 第一次 block 可能 downsample
        blocks.append(ResBlock1D(in_ch, out_ch, stride=stride))
        # 其餘 block stride=1
        for _ in range(num_blocks - 1):
            blocks.append(ResBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_in(x)   # => (B, base_filters, ~256)
        x = self.stage1(x)    # => (B, base_filters, ~256)
        x = self.stage2(x)    # => (B, base_filters*2, ~128)
        x = self.stage3(x)    # => (B, base_filters*4, ~64)
        x = self.stage4(x)    # => (B, base_filters*8, ~32)
        x = self.global_pool(x)  # => (B, base_filters*8, 1)
        x = x.squeeze(-1)        # => (B, base_filters*8)
        return x


######################################
# 3) 主模型: BPEstimator
######################################

class BPEstimator(nn.Module):
    """
    超深ResNet分支 (DeepResBranchNet) x 4 + 個人資訊 => 較大 MLP => 預測2維
    """
    def __init__(self, info_dim=5, base_filters=64, layers_per_stage=[3,4,6,3]):
        super().__init__()
        # 四分支
        self.ppg_branch = DeepResBranchNet(in_channels=1, base_filters=base_filters, layers_per_stage=layers_per_stage)
        self.vpg_branch = DeepResBranchNet(in_channels=1, base_filters=base_filters, layers_per_stage=layers_per_stage)
        self.apg_branch = DeepResBranchNet(in_channels=1, base_filters=base_filters, layers_per_stage=layers_per_stage)
        self.ecg_branch = DeepResBranchNet(in_channels=1, base_filters=base_filters, layers_per_stage=layers_per_stage)

        # out_ch = base_filters*8
        self.signal_feat_dim = base_filters * 8 * 4  # 四路 => x4

        self.info_fc = nn.Sequential(
            nn.Linear(info_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # => (B,64)

        # 最後 MLP
        # e.g. (base_filters*8*4 + 64) => 3200~? => 1024 => ...
        self.final_fc = nn.Sequential(
            nn.Linear(self.signal_feat_dim + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, ppg, vpg, apg, ecg, personal_info):
        ppg_feat = self.ppg_branch(ppg)
        vpg_feat = self.vpg_branch(vpg)
        apg_feat = self.apg_branch(apg)
        ecg_feat = self.ecg_branch(ecg)
        signal_feat = torch.cat([ppg_feat, vpg_feat, apg_feat, ecg_feat], dim=1)

        info_feat = self.info_fc(personal_info)
        combined = torch.cat([signal_feat, info_feat], dim=1)
        out = self.final_fc(combined)
        return out

######################################
# 4) 訓練 / 驗證 / 測試流程
######################################
class BPTrainer:
    def __init__(self, fold_path, device='cuda', batch_size=32, lr=1e-3, info_dim=5,
                 base_filters=32, num_blocks=4):
        """
        fold_path: e.g. 'training_data/'
        info_dim : personal_info 維度
        base_filters, num_blocks: DeepResBranchNet 相關超參
        """
        self.fold_path = Path(fold_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr
        self.info_dim = info_dim
        self.base_filters = base_filters
        self.num_blocks = num_blocks

    def create_dataloaders(self):
        # 1) 收集所有 training_x.h5
        train_files = [self.fold_path / f"training_{i}.h5" for i in range(1, 10)]
        
        # 2) 建立一個 MultiH5Dataset
        train_set = MultiH5Dataset(train_files)  # 會把 9 個 .h5 的資料串起來

        # Validation, Test 檔案
        val_set   = BPDataset(self.fold_path / 'validation.h5')
        test_set  = BPDataset(self.fold_path / 'test.h5')

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=self.batch_size, shuffle=False, drop_last=False)

        # (檢查 batch)
        sample_batch = next(iter(train_loader))
        print("\n[檢查一個 train batch]")
        for k, v in sample_batch.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        return train_loader, val_loader, test_loader

    def visualize_model(self, model, output_path="model_structure.png"):
        """
        使用 torchviz 生成模型架構圖並保存為圖片
        """
        print("[INFO] 正在生成模型架構圖...")
        # 創建一組假資料
        dummy_ppg = torch.randn(1, 1, 1024).to(self.device)
        dummy_vpg = torch.randn(1, 1, 1024).to(self.device)
        dummy_apg = torch.randn(1, 1, 1024).to(self.device)
        dummy_ecg = torch.randn(1, 1, 1024).to(self.device)
        dummy_info = torch.randn(1, self.info_dim).to(self.device)

        # 前向傳播，獲取模型輸出
        output = model(dummy_ppg, dummy_vpg, dummy_apg, dummy_ecg, dummy_info)

        # 使用 make_dot 生成模型圖
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render(output_path, cleanup=True)
        print(f"[INFO] 模型架構圖已保存至 {output_path}")

    def train(self, epochs=50, early_stop_patience=10):

        # 建構模型
        model = BPEstimator(
            info_dim=self.info_dim,
            base_filters=self.base_filters,
            # num_blocks=self.num_blocks
        ).to(self.device)
        # 印出參數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Model] total_params = {total_params:,}, trainable_params = {trainable_params:,}\n")

        # 可視化模型架構
        # self.visualize_model(model, output_path="model_structure")
        input("Press Enter to continue...")
        train_loader, val_loader, test_loader = self.create_dataloaders()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)  
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)


        # ======================
        # 可選的 torchviz 可視化程式碼已移到 visualize_model 方法
        # ======================

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs+1):
            # ---------- Training ----------
            model.train()
            train_loss = 0.0
            train_mae  = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)"):
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values = batch['bp_values'].to(self.device)      # (B,2)
                personal_info = batch['personal_info'].to(self.device)

                optimizer.zero_grad()
                preds = model(ppg, vpg, apg, ecg, personal_info)
                loss = criterion(preds, bp_values)
                mae  = torch.mean(torch.abs(preds - bp_values))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mae  += mae.item()

            train_loss /= len(train_loader)
            train_mae  /= len(train_loader)

            # ---------- Validation ----------
            model.eval()
            val_loss = 0.0
            val_mae  = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    ppg = batch['ppg'].to(self.device)
                    vpg = batch['vpg'].to(self.device)
                    apg = batch['apg'].to(self.device)
                    ecg = batch['ecg'].to(self.device)
                    bp_values = batch['bp_values'].to(self.device)
                    personal_info = batch['personal_info'].to(self.device)

                    preds = model(ppg, vpg, apg, ecg, personal_info)
                    loss = criterion(preds, bp_values)
                    mae  = torch.mean(torch.abs(preds - bp_values))

                    val_loss += loss.item()
                    val_mae  += mae.item()

            val_loss /= len(val_loader)
            val_mae  /= len(val_loader)

            # learning rate scheduler
            scheduler.step(val_loss)

            # EarlyStopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.fold_path / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("[Info] Early stopping triggered.")
                    break

            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss={train_loss:.4f}, MAE={train_mae:.4f} | "
                  f"Val Loss={val_loss:.4f}, MAE={val_mae:.4f}")

        # ---------- 測試 ----------
        print("\n[測試階段] Load best_model.pt")
        model.load_state_dict(torch.load(self.fold_path / 'best_model.pt'))
        model.eval()
        test_loss = 0.0
        test_mae  = 0.0
        with torch.no_grad():
            for batch in test_loader:
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values = batch['bp_values'].to(self.device)
                personal_info = batch['personal_info'].to(self.device)

                preds = model(ppg, vpg, apg, ecg, personal_info)
                loss = criterion(preds, bp_values)
                mae  = torch.mean(torch.abs(preds - bp_values))

                test_loss += loss.item()
                test_mae  += mae.item()

        test_loss /= len(test_loader)
        test_mae  /= len(test_loader)
        print(f"[Test Results] MSE={test_loss:.4f}, MAE={test_mae:.4f}")

        return model


######################################
# 5) 主程式
######################################
if __name__ == '__main__':
    """
    假設資料夾結構:
      training_data/
        train.h5
        val.h5
        test.h5
    執行:
      python bp_model_trainer.py
    即可進行訓練 + 測試，並輸出 best_model.pt
    """
    trainer = BPTrainer(
        fold_path='training_data',
        device='cuda',
        batch_size=64,
        lr=1e-3,
        info_dim=5,     # personal_info 維度
        base_filters=32, 
        num_blocks=4    # ResBlock 疊多少層
    )
    final_model = trainer.train(epochs=50, early_stop_patience=10)
