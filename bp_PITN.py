#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面優化版的 PITN 模型：
結合物理約束、對抗訓練與對比學習，
並支持 few-shot personalized 微調，
主要針對效能瓶頸進行以下改進：
  - 模型維度與 batch size 調小
  - 使用混合精度訓練
  - 在 PITN.forward 中返回全局特徵，避免重複計算
  - PGD 攻擊中關閉二階梯度計算
  - Windows 平台下 DataLoader 的 num_workers 設為 0 以避免 pickle 錯誤
"""

import os
import math
import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from tqdm import tqdm

#############################################
# 1. 數據集類：VitalSignDataset
#############################################
def extract_features_from_annotations(anno_matrix):
    """
    從 anno_matrix (1250,4) 中提取三個生理特徵：
      feat1：ECG_RealPeaks 間的平均間隔
      feat2：PPG_Turns 到 ECG_RealPeaks 的平均時間差
      feat3：PPG_SPeaks 到 ECG_RealPeaks 的平均時間差
    """
    indices_ecg = np.where(anno_matrix[:, 0] == 1)[0]
    indices_ppg_s = np.where(anno_matrix[:, 1] == 1)[0]
    indices_ppg_t = np.where(anno_matrix[:, 2] == 1)[0]
    feat1 = np.mean(np.diff(indices_ecg)) if len(indices_ecg) > 1 else 0
    diffs2 = [peak - indices_ppg_t[indices_ppg_t <= peak][-1]
              for peak in indices_ecg if len(indices_ppg_t[indices_ppg_t <= peak]) > 0]
    feat2 = np.mean(diffs2) if len(diffs2) > 0 else 0
    diffs3 = [peak - indices_ppg_s[indices_ppg_s <= peak][-1]
              for peak in indices_ecg if len(indices_ppg_s[indices_ppg_s <= peak]) > 0]
    feat3 = np.mean(diffs3) if len(diffs3) > 0 else 0
    return feat1, feat2, feat3

class VitalSignDataset(Dataset):
    """
    從 .h5 文件中讀取：
      - ppg: (N,1250)
      - ecg: (N,1250)
      - abp: (N,1250)  (輔助觀察)
      - annotations: (N,1250,4)
      - segsbp: (N,)
      - segdbp: (N,)
      - personal_info: (N,4)
    並根據 annotations 提取出 3 維生理特徵 (phys_feat)；
    與 personal_info 拼接後 extra_features 維度 = 3 + 4 = 7。
    """
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            # 讀取原始數據，並轉成 float32
            ppg_np = f['ppg'][:].astype(np.float32)       # (N,1250)
            ecg_np = f['ecg'][:].astype(np.float32)       # (N,1250)
            abp_np = f['abp'][:].astype(np.float32)       # (N,1250)
            segsbp_np = f['segsbp'][:].astype(np.float32)   # (N,)
            segdbp_np = f['segdbp'][:].astype(np.float32)   # (N,)
            annotations_np = f['annotations'][:]          # (N,1250,4)
            if 'personal_info' in f:
                personal_info_np = f['personal_info'][:].astype(np.float32)  # (N,4)
            else:
                personal_info_np = np.zeros((ppg_np.shape[0], 4), dtype=np.float32)
        
        # 進行全局 min-max 正規化到 [0, 1]
        ppg_min, ppg_max = ppg_np.min(), ppg_np.max()
        ecg_min, ecg_max = ecg_np.min(), ecg_np.max()
        abp_min, abp_max = abp_np.min(), abp_np.max()
        ppg_np = (ppg_np - ppg_min) / (ppg_max - ppg_min + 1e-8)
        ecg_np = (ecg_np - ecg_min) / (ecg_max - ecg_min + 1e-8)
        abp_np = (abp_np - abp_min) / (abp_max - abp_min + 1e-8)
        
        self.ppg = torch.from_numpy(ppg_np)
        self.ecg = torch.from_numpy(ecg_np)
        self.abp = torch.from_numpy(abp_np)
        self.segsbp = torch.from_numpy(segsbp_np)
        self.segdbp = torch.from_numpy(segdbp_np)
        self.annotations = annotations_np
        self.personal_info = torch.from_numpy(personal_info_np)
        self.n_samples = self.ppg.shape[0]
        # 將 ppg, ecg 增加 channel 維度 → (N,1,1250)
        self.ppg = self.ppg.unsqueeze(1)
        self.ecg = self.ecg.unsqueeze(1)
        # 組成 bp_values (N,2)
        self.bp_values = torch.stack([self.segsbp, self.segdbp], dim=1)
        # 從 annotations 提取 3 維生理特徵
        phys_feats = []
        for i in range(self.n_samples):
            anno = self.annotations[i]
            feat1, feat2, feat3 = extract_features_from_annotations(anno)
            phys_feats.append([feat1, feat2, feat3])
        self.phys_feat = torch.tensor(phys_feats).float()
        self.demo_feat = self.personal_info
        self.extra_features = torch.cat([self.phys_feat, self.demo_feat], dim=1)  # (N,7)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx],
            'ecg': self.ecg[idx],
            'bp_values': self.bp_values[idx],
            'extra_features': self.extra_features[idx],
            'phys_feat': self.phys_feat[idx]
        }

#############################################
# 2. 模型模塊
#############################################
# 2.1 DataEmbedding 模塊
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        return self.tokenConv(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.pos_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.value_embedding(x)  # (B, d_model, T)
        x = x.transpose(1,2) + self.pos_embedding(x.transpose(1,2))
        return self.dropout(x)  # (B, T, d_model)

# 2.2 TemporalBlock 模塊
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super(Inception_Block_V1, self).__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i)
            for i in range(num_kernels)
        ])
    def forward(self, x):
        outs = [kernel(x) for kernel in self.kernels]
        return torch.stack(outs, dim=-1).mean(-1)

def FFT_for_Period(x, top_k=1):
    x_fft = torch.fft.rfft(x, dim=1)
    amp = torch.abs(x_fft).mean(dim=0).mean(dim=-1)
    amp[0] = 0
    top_vals, top_idx = torch.topk(amp, top_k)
    period = x.shape[1] // top_idx[0].item() if top_idx[0].item() != 0 else x.shape[1]
    return period

class TemporalBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_kernels=6):
        super(TemporalBlock, self).__init__()
        self.inception = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )
    def forward(self, x):
        B, T, D = x.size()
        period = FFT_for_Period(x, top_k=1)
        if T % period != 0:
            pad_len = period - (T % period)
            pad = torch.zeros(B, pad_len, D, device=x.device)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x
        new_T = x_padded.size(1)
        num_rows = new_T // period
        x_2d = x_padded.reshape(B, num_rows, period, D).permute(0, 3, 1, 2)
        out = self.inception(x_2d)
        out = out.permute(0, 2, 3, 1).reshape(B, new_T, D)
        out = out[:, :T, :]
        return out + x

# 2.3 PITN 模型
class PITN(nn.Module):
    def __init__(self, in_channels=2, d_model=32, d_ff=64, n_layers=2, dropout=0.1, extra_dim=7, out_dim=2):
        super(PITN, self).__init__()
        self.embedding = DataEmbedding(in_channels, d_model, dropout=dropout)
        self.temporal_blocks = nn.ModuleList([TemporalBlock(d_model, d_ff) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model + extra_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, waveform, extra, return_features=False):
        x = self.embedding(waveform)  # (B, T, d_model)
        for block in self.temporal_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x_trans = x.transpose(1,2)  # (B, d_model, T)
        global_feat = self.global_pool(x_trans).squeeze(-1)  # (B, d_model)
        x_cat = torch.cat([global_feat, extra], dim=1)
        out = self.fc(x_cat)
        if return_features:
            return out, global_feat
        else:
            return out
    def physics_loss(self, waveform, phys_feat):
        waveform.requires_grad_(True)
        phys_feat.requires_grad_(True)
        dummy_demo = torch.zeros(phys_feat.size(0), 4, device=phys_feat.device)
        extra = torch.cat([phys_feat, dummy_demo], dim=1)
        u = self.forward(waveform, extra)
        u = u[:, 0]
        u_hat = []
        for i in range(u.size(0) - 1):
            ui = u[i]
            grad_ui = torch.autograd.grad(ui, phys_feat[i], retain_graph=True, create_graph=True, allow_unused=True)[0]
            if grad_ui is None:
                grad_ui = torch.zeros_like(phys_feat[i])
            u_hat_i = ui + torch.dot(grad_ui, phys_feat[i+1] - phys_feat[i])
            u_hat.append(u_hat_i)
        u_hat = torch.stack(u_hat)
        target = u[1:]
        return F.mse_loss(u_hat, target)

#############################################
# 3. 對抗訓練 (PGD) 與 對比學習
#############################################
class regression_PGD:
    def __init__(self, model, lb, ub, eps=0.2, eta=0.02, steps=2, loss=nn.L1Loss()):
        self.model = model
        self.eps = eps
        self.eta = eta
        self.steps = steps
        self.loss = loss
        self.lb = lb
        self.ub = ub
        self.device = next(model.parameters()).device
    def attack(self, samples, targets, extra):
        if not torch.is_tensor(samples):
            samples = torch.tensor(samples, dtype=torch.float32).to(self.device)
        else:
            samples = samples.clone().detach().to(self.device)
        adv_samples = samples.clone().detach()
        adv_samples += torch.empty_like(adv_samples).uniform_(-self.eps, self.eps)
        for i in range(adv_samples.shape[1]):
            adv_samples[:, i, :] = torch.clamp(adv_samples[:, i, :], min=self.lb[i], max=self.ub[i])
        adv_samples = adv_samples.detach()
        for _ in range(self.steps):
            adv_samples.requires_grad = True
            f = self.model(adv_samples, extra)
            loss_val = self.loss(f, targets)
            grad = torch.autograd.grad(loss_val, adv_samples,
                                       grad_outputs=torch.ones_like(loss_val),
                                       retain_graph=False, create_graph=False)[0]
            adv_samples = adv_samples.detach() + self.eta * grad.sign()
            delta = torch.clamp(adv_samples - samples, min=-self.eps, max=self.eps)
            adv_samples = samples + delta
            for i in range(adv_samples.shape[1]):
                adv_samples[:, i, :] = torch.clamp(adv_samples[:, i, :], min=self.lb[i], max=self.ub[i])
            adv_samples = adv_samples.detach()
        return adv_samples

class MultiPosConLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=-1)
        logits = torch.matmul(feats, feats.T) / self.temperature
        threshold = 2.0
        labels = labels.unsqueeze(1)
        diff = torch.abs(labels - labels.T)
        mask = (diff < threshold).float()
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        p = mask / mask_sum
        log_prob = F.log_softmax(logits, dim=1)
        return - (p * log_prob).sum(dim=1).mean()

#############################################
# 4. 綜合訓練流程
#############################################
def train_model_full(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float('inf')
    lb = [0.0, 0.0]  # 假設輸入 ppg 與 ecg 維度均在 [0,1]
    ub = [1.0, 1.0]
    pgd = regression_PGD(model, lb, ub, eps=0.2, eta=0.02, steps=2, loss=mse_loss)
    contrastive_loss_fn = MultiPosConLoss(temperature=1.0)
    gamma_phys = 1.0
    alpha_con = 1.0
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep} Training", leave=False):
            waveform = batch['ppg'].to(device)
            ecg = batch['ecg'].to(device)
            # 融合兩路信號
            waveform = torch.cat([waveform, ecg], dim=1)  # (B,2,T)
            extra = batch['extra_features'].to(device)      # (B,7)
            bp = batch['bp_values'].to(device)              # (B,2)
            phys_feat = batch['phys_feat'].to(device)       # (B,3)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds_clean, global_feat = model(waveform, extra, return_features=True)
                loss_clean = mse_loss(preds_clean, bp)
                adv_waveform = pgd.attack(waveform, bp, extra)
                preds_adv = model(adv_waveform, extra)
                loss_adv = mse_loss(preds_adv, bp)
                labels_con = preds_clean[:, 0]
                loss_con = contrastive_loss_fn(global_feat, labels_con)
                loss_phys = model.physics_loss(waveform, phys_feat) if bp.size(0) > 1 else 0.0
                total_loss = loss_clean + loss_adv + alpha_con * loss_con + gamma_phys * loss_phys
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += total_loss.item()
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch['ppg'].to(device)
                ecg = batch['ecg'].to(device)
                waveform = torch.cat([waveform, ecg], dim=1)
                extra = batch['extra_features'].to(device)
                bp = batch['bp_values'].to(device)
                preds = model(waveform, extra)
                loss = mse_loss(preds, bp)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {ep}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_state = model.state_dict()
            torch.save(best_state, os.path.join("pitn_pretrained_full.pth"))
        torch.cuda.empty_cache()
    model.load_state_dict(best_state)
    return model

def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for batch in test_loader:
            waveform = batch['ppg'].to(device)
            ecg = batch['ecg'].to(device)
            waveform = torch.cat([waveform, ecg], dim=1)
            extra = batch['extra_features'].to(device)
            bp = batch['bp_values'].to(device)
            preds = model(waveform, extra)
            loss = mse_loss(preds, bp)
            mae = torch.mean(torch.abs(preds - bp))
            total_loss += loss.item()
            total_mae += mae.item()
    avg_loss = total_loss / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    print(f"Test MSE={avg_loss:.4f}, MAE={avg_mae:.4f}")

#############################################
# 5. Personalized Few-shot 微調流程
#############################################
def fine_tune_personalized(model, personal_dataset, epochs=30, lr=1e-4, device='cuda'):
    personal_loader = DataLoader(personal_dataset, batch_size=8, shuffle=True, num_workers=0)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_loss = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in personal_loader:
            waveform = batch['ppg'].to(device)
            ecg = batch['ecg'].to(device)
            waveform = torch.cat([waveform, ecg], dim=1)
            extra = batch['extra_features'].to(device)
            bp = batch['bp_values'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(waveform, extra), bp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(personal_loader)
        print(f"[Personalized Fine-tuning] Epoch {ep}: Loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model

#############################################
# 6. 主流程
#############################################
def main():
    data_dir = Path('training_data_VitalDB_quality')
    train_files = [data_dir / f"training_{i+1}.h5" for i in range(9)]
    val_file = data_dir / 'validation.h5'
    test_file = data_dir / 'test.h5'
    
    # 建立數據集
    train_dss = []
    for tf in train_files:
        if tf.exists():
            train_dss.append(VitalSignDataset(str(tf)))
        else:
            print(f"Warning: Training file {tf} not found")
    if not train_dss:
        raise FileNotFoundError("No training files found.")
    train_dataset = ConcatDataset(train_dss)
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found at {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found at {test_file}")
    val_dataset = VitalSignDataset(str(val_file))
    test_dataset = VitalSignDataset(str(test_file))
    
    # 為避免 Windows 平台多進程 pickle 問題，num_workers 設為 0
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    
    print(f"Length of train_loader: {len(train_loader)}")
    print(f"Length of val_loader: {len(val_loader)}")
    print(f"Length of test_loader: {len(test_loader)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PITN(in_channels=2, d_model=32, d_ff=64, n_layers=2, dropout=0.1, extra_dim=7, out_dim=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("Start Pre-training PITN on training_data_VitalDB_quality")
    model = train_model_full(model, train_loader, val_loader, epochs=50, lr=1e-3, device=device)
    test_model(model, test_loader, device=device)
    
    pretrained_model_path = "pitn_pretrained_full.pth"
    torch.save(model.state_dict(), pretrained_model_path)
    print(f"Pretrained model saved to {pretrained_model_path}")
    
    personal_data_dir = Path("personalized_training_data_VitalDB")
    if personal_data_dir.exists():
        personal_files = list(personal_data_dir.glob("*.h5"))
        for pf in personal_files:
            print(f"Fine-tuning for {pf.name}")
            personal_dataset = VitalSignDataset(str(pf))
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            model = fine_tune_personalized(model, personal_dataset, epochs=30, lr=1e-4, device=device)
            torch.save(model.state_dict(), f"pitn_personalized_{pf.stem}.pth")
    else:
        print(f"Personal data directory not found at {personal_data_dir}")

if __name__ == "__main__":
    main()
