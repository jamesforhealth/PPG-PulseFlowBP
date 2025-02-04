import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import random
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

#############################################
# (A) GlobalABPDataset：根據 H5 檔案讀取資料
#############################################
class GlobalABPDataset(Dataset):
    """
    將多個 .h5 檔案 (包含 ppg, ecg, abp, personal_info, vascular, segsbp, segdbp)
    合併成單一 Dataset。
    
    輸出：
      - wave_input: (2,1250) 由 [ppg, ecg] 組成
      - wave_target: (1250) 來自 abp 波形
      - extra: (6,) 由 [personal_info (4), vascular_properties (2)] 組成
      - sbp, dbp: 單一 float (用於計算峰值/谷值 MAE)
    """
    def __init__(self, h5_files):
        self.samples = []
        for hf in h5_files:
            with h5py.File(hf, 'r') as f:
                ppg_all = f['ppg'][:]    # (N,1250)
                ecg_all = f['ecg'][:]
                abp_all = f['abp'][:]
                pers_all = f['personal_info'][:]
                vasc_all = f['vascular_properties'][:]
                sbp_all = f['segsbp'][:]  # (N,)
                dbp_all = f['segdbp'][:]  # (N,)
                N = len(ppg_all)
                for i in range(N):
                    data_item = {
                        'ppg': ppg_all[i],
                        'ecg': ecg_all[i],
                        'abp': abp_all[i],
                        'personal': pers_all[i],
                        'vascular': vasc_all[i],
                        'sbp': sbp_all[i],
                        'dbp': dbp_all[i],
                    }
                    self.samples.append(data_item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        ppg_ = s['ppg']
        ecg_ = s['ecg']
        wave_2ch = np.stack([ppg_, ecg_], axis=0)  # (2,1250)
        abp_ = s['abp']
        pers_ = s['personal']
        vasc_ = s['vascular']
        extra_6 = np.concatenate([pers_, vasc_], axis=0)
        sbp_ = s['sbp']
        dbp_ = s['dbp']
        wave_t = torch.from_numpy(wave_2ch).float()
        abp_t = torch.from_numpy(abp_).float()
        extra_t = torch.from_numpy(extra_6).float()
        sbp_val = torch.tensor(sbp_, dtype=torch.float32)
        dbp_val = torch.tensor(dbp_, dtype=torch.float32)
        return wave_t, abp_t, extra_t, sbp_val, dbp_val

#############################################
# (B) Positional Encoding
#############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

#############################################
# (C) PatchTST 模型
#############################################
class PatchTSTModel(nn.Module):
    """
    將 2x1250 輸入分割成 patch，經線性投影、位置編碼、Transformer Encoder 處理，
    最後重構回 1250 長度的預測。
    """
    def __init__(self, patch_size=25, d_model=64, num_encoder_layers=4, nhead=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = 1250 // patch_size
        self.input_dim = 2 * patch_size
        
        self.patch_embed = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.patch_reconstruct = nn.Linear(d_model, patch_size)
    
    def forward(self, wave):
        B, C, L = wave.shape  # (B,2,1250)
        patches = wave.unfold(dimension=2, size=self.patch_size, step=self.patch_size)  # (B,2, num_patches, patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, self.num_patches, -1)  # (B, num_patches, 2*patch_size)
        x = self.patch_embed(patches)  # (B, num_patches, d_model)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)  # (B, num_patches, d_model)
        patch_out = self.patch_reconstruct(x)  # (B, num_patches, patch_size)
        output = patch_out.view(B, -1)  # (B,1250)
        return output

#############################################
# (D) Reformer 模型包裝器
#############################################
# 請先安裝 reformer-pytorch： pip install reformer-pytorch
try:
    from reformer_pytorch import Reformer
except ImportError:
    raise ImportError("請先安裝 reformer-pytorch: pip install reformer-pytorch")

class ReformerModelWrapper(nn.Module):
    """
    使用 reformer-pytorch 實現 Reformer 模型：
      將 2×1250 輸入轉換為 (B,1250,2)，線性升維到 d_model，
      經過 Reformer 模型後，再用線性層映射到 1 維預測 ABP。
    """
    def __init__(self, d_model=64, depth=4, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(2, d_model)
        self.bucket_size = 64  # 我們仍然使用預設值 64
        self.reformer = Reformer(
            dim=d_model,
            depth=depth,
            heads=nhead,
            causal=False,
            bucket_size=self.bucket_size,  # 注意：此處不再傳入 dropout
            n_hashes=4
        )
        self.fc_out = nn.Linear(d_model, 1)
    
    def forward(self, wave):
        # wave: (B,2,1250) -> (B,1250,2)
        x = wave.transpose(1, 2)  # (B, L, 2)
        L = x.size(1)
        divisor = self.bucket_size * 2  # 要求序列長度能被 bucket_size*2 整除
        pad_len = (divisor - (L % divisor)) if L % divisor != 0 else 0
        if pad_len:
            # pad 右側 (在時間軸上)
            padding = torch.zeros(x.size(0), pad_len, x.size(2), device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        x = self.input_projection(x)  # (B, L+pad_len, d_model)
        x = self.reformer(x)          # (B, L+pad_len, d_model)
        x = self.fc_out(x)            # (B, L+pad_len, 1)
        x = x.squeeze(-1)             # (B, L+pad_len)
        if pad_len:
            x = x[:, :L]  # 裁剪回原長度
        return x

#############################################
# (E) Loss Function
#############################################
def wave_loss_with_peak_valley(pred, target, alpha=0.1, threshold=0.5):
    mae_func = nn.SmoothL1Loss(beta=threshold)
    base_loss = mae_func(pred, target)
    pred_max, _ = torch.max(pred, dim=1)
    targ_max, _ = torch.max(target, dim=1)
    pred_min, _ = torch.min(pred, dim=1)
    targ_min, _ = torch.min(target, dim=1)
    peak_err = torch.abs(pred_max - targ_max).mean()
    valley_err = torch.abs(pred_min - targ_min).mean()
    penalty = alpha * (peak_err + valley_err)
    return base_loss + penalty

#############################################
# (F) 繪圖函數：從 validation 中抽樣並繪圖
#############################################
def plot_val_samples(model, val_dataset, device, num_samples=3, epoch=None):
    model.eval()
    indices = random.sample(range(len(val_dataset)), num_samples)
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            wave, abp, extra, sbp, dbp = val_dataset[idx]
            wave = wave.unsqueeze(0).to(device)
            extra = extra.unsqueeze(0).to(device)
            # 對於部分模型，若模型需要 extra 輸入則傳入
            try:
                pred = model(wave, extra)
            except TypeError:
                pred = model(wave)
            pred = pred.squeeze(0).cpu().numpy()
            true_abp = abp.cpu().numpy()
            x_axis = np.arange(len(true_abp))
            axs[i, 0].plot(x_axis, true_abp, color='blue')
            axs[i, 0].set_title("True ABP")
            axs[i, 1].plot(x_axis, pred, color='red')
            axs[i, 1].set_title("Predicted ABP")
            axs[i, 2].plot(x_axis, true_abp, label="True ABP", color='blue')
            axs[i, 2].plot(x_axis, pred, label="Predicted ABP", color='red', linestyle='--')
            axs[i, 2].set_title("Overlay")
            axs[i, 2].legend()
    if epoch is not None:
        fig.suptitle(f"Validation Samples at Epoch {epoch}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#############################################
# (G) 訓練與評估函數
#############################################
def train_model(model, train_loader, val_loader, epochs=5, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    best_state = None
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        count = 0
        for wave, abp, extra, sbp, dbp in tqdm(train_loader, desc=f"Training Epoch {ep}", leave=False):
            wave, abp = wave.to(device), abp.to(device)
            optimizer.zero_grad()
            try:
                output = model(wave, extra.to(device))
            except TypeError:
                output = model(wave)
            loss = wave_loss_with_peak_valley(output, abp, alpha=0.1, threshold=0.5)
            loss.backward()
            optimizer.step()
            bs = wave.size(0)
            total_loss += loss.item() * bs
            count += bs
        train_loss = total_loss / count
        
        model.eval()
        v_total = 0.0
        v_count = 0
        with torch.no_grad():
            for wave, abp, extra, sbp, dbp in tqdm(val_loader, desc=f"Validation Epoch {ep}", leave=False):
                wave, abp = wave.to(device), abp.to(device)
                try:
                    output = model(wave, extra.to(device))
                except TypeError:
                    output = model(wave)
                loss = wave_loss_with_peak_valley(output, abp, alpha=0.1, threshold=0.5)
                bs = wave.size(0)
                v_total += loss.item() * bs
                v_count += bs
        val_loss = v_total / v_count
        print(f"Epoch {ep}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
        plot_val_samples(model, val_loader.dataset, device, num_samples=3, epoch=ep)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for wave, abp, extra, sbp, dbp in test_loader:
            wave, abp = wave.to(device), abp.to(device)
            try:
                output = model(wave, extra.to(device))
            except TypeError:
                output = model(wave)
            loss = wave_loss_with_peak_valley(output, abp, alpha=0.1, threshold=0.5)
            bs = wave.size(0)
            total_loss += loss.item() * bs
            count += bs
    test_loss = total_loss / count
    print(f"Test Loss = {test_loss:.4f}")
    return test_loss

#############################################
# (H) 主函數：比較 PatchTST 與 Reformer 模型
#############################################
def main():
    # 請根據實際情況修改 H5 檔案路徑
    data_dir = Path('training_data_VitalDB_quality')
    train_files = [data_dir / f"training_{i+1}.h5" for i in range(9)]
    val_file = data_dir / 'validation.h5'
    test_file = data_dir / 'test.h5'
    
    train_dataset = GlobalABPDataset(train_files)
    val_dataset = GlobalABPDataset([val_file])
    test_dataset = GlobalABPDataset([test_file])
    
    print(f"Train set size= {len(train_dataset)}, Val= {len(val_dataset)}, Test= {len(test_dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------
    # 1. PatchTST 模型訓練與評估
    # -------------------------------
    # print("==== Training PatchTSTModel ====")
    # patchtst_model = PatchTSTModel(patch_size=25, d_model=64, num_encoder_layers=4, nhead=4, dropout=0.1)
    # patchtst_loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # patchtst_loader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # patchtst_loader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # patchtst_model = train_model(patchtst_model, patchtst_loader_train, patchtst_loader_val, epochs=5, device=device)
    # print("==== Evaluating PatchTSTModel ====")
    # evaluate_model(patchtst_model, patchtst_loader_test, device=device)
    
    # -------------------------------
    # 2. Reformer 模型訓練與評估
    # -------------------------------
    print("\n==== Training ReformerModelWrapper ====")
    reformer_model = ReformerModelWrapper(d_model=64, depth=4, nhead=4)
    # sum of paran
    print(f"ReformerModelWrapper parameters: {sum(p.numel() for p in reformer_model.parameters())}")
    reformer_loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    reformer_loader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
    reformer_loader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    reformer_model = train_model(reformer_model, reformer_loader_train, reformer_loader_val, epochs=5, device=device)
    print("==== Evaluating ReformerModelWrapper ====")
    evaluate_model(reformer_model, reformer_loader_test, device=device)
    
    # 可根據需求保存模型權重
    # torch.save(patchtst_model.state_dict(), "patchtst_wave2wave.pth")
    torch.save(reformer_model.state_dict(), "reformer_wave2wave.pth")
    print("Done.")

if __name__ == "__main__":
    main()
