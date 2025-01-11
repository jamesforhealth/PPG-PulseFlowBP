import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
import math
from torch.nn.modules.loss import HuberLoss
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

def load_batch(file_path):
    return torch.load(file_path, weights_only=True, map_location=torch.device('cpu'))
class BatchDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        pair = load_batch(file_path)
        # print(f'pair len: {len(pair)}')
        ppg = pair[0]
        bp = pair[1]
        return ppg, bp

def custom_collate_fn(batch):
    # batch 是一個包含多個 __getitem__ 返回值的列表
    ppgs, bps = zip(*batch)
    
    # ppgs 和 bps 是 tuple，通常你希望將它們拼接成一個批次
    # 但在這裡，你可以直接返回列表中的第一個元素，避免增加維度
    ppgs = ppgs[0]  # [128, 1250]
    bps = bps[0]    # [58]，假設目標序列長度已經填充到 58
    
    return ppgs, bps

class PPGtoBPTransformer(nn.Module):
    def __init__(self, input_dim=1250, output_dim=58, d_model=64, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.3):
        super(PPGtoBPTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_proj(src)  # [seq_len, d_model]
        src = src.unsqueeze(1)  # [seq_len, 1, d_model]
        src = self.positional_encoding(src)
        
        output = self.transformer_encoder(src)
        output = self.output_proj(output)
        output = output.squeeze(1)
        return output  # [batch_size, 58]



def train_model(model, train_loader, val_loader, model_path, epochs=10, lr=1e-5, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = HuberLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for ppg, bp in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            ppg, bp = ppg.to(device), bp.to(device)
            optimizer.zero_grad()
            # input(f'ppg: {ppg.shape}, bp: {bp.shape}')
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(ppg)
                loss = criterion(output, bp)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            score = loss.item()
            # print(f'batch loss: {score}')
            #if inf
            if score == float('inf'):
                # print(f'ppg: {ppg}')
                print(f'bp: {bp.min()}')
                print(f'bp: {bp.max()}')
                print(f'output: {output.min()}')
                print(f'output: {output.max()}')
                print(f'bp: {bp}')
                input(f'output: {output}')

            train_loss += score
            

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ppg, bp in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                ppg, bp = ppg.to(device), bp.to(device)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(ppg)
                    loss = criterion(output, bp)
                
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss * 0.95:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.8f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

    return model

def evaluate(model, test_files, h5_folder, device='cuda'):
    model.eval()
    mae_f_sbp, mae_f_dbp = 0, 0
    mae_raw_sbp, mae_raw_dbp = 0, 0
    total_samples = 0

    with torch.no_grad():
        for filename in tqdm(test_files, desc="Evaluating"):
            h5_path = os.path.join(h5_folder, filename)
            with h5py.File(h5_path, 'r') as f:
                ppg_raw = f['PPG_Raw'][:]
                abp_f = f['ABP_F'][:]
                abp_speaks = f['ABP_SPeaks'][:]
                abp_turns = f['ABP_Turns'][:]
                abp_raw = f['ABP_Raw'][:]

            for i in range(len(ppg_raw)):
                ppg = torch.FloatTensor(ppg_raw[i]).unsqueeze(0).to(device)  # [1, 1250]
                abp_f_true = torch.FloatTensor(abp_f[i, 0])
                speaks = abp_speaks[i]
                turns = abp_turns[i]

                # 预测
                pred_f = model(ppg).squeeze(0).cpu()  # [58]
                # print(f'pred_f: {pred_f}')

                # 获取 SBP 和 DBP 点的索引
                sbp_points = speaks - 1
                dbp_points = turns - 1

                # 合并并排序 SBP 和 DBP 点
                all_points = np.sort(np.concatenate([sbp_points, dbp_points]))

                # 确保预测值和真实值的长度匹配
                min_len = min(len(pred_f), len(all_points))
                pred_f = pred_f[:min_len]
                all_points = all_points[:min_len]

                # 分离 SBP 和 DBP 预测值
                pred_sbp = pred_f[np.isin(all_points, sbp_points)]
                pred_dbp = pred_f[np.isin(all_points, dbp_points)]
                # print(f'pred_sbp: {pred_sbp}')
                # print(f'pred_dbp: {pred_dbp}')


                # 计算 ABP_F 的 MAE
                mae_f_sbp += torch.abs(pred_sbp - abp_f_true[sbp_points[:len(pred_sbp)]]).mean().item()
                mae_f_dbp += torch.abs(pred_dbp - abp_f_true[dbp_points[:len(pred_dbp)]]).mean().item()

                # 线性缩放回 ABP_RAW
                abp_raw_true = abp_raw[i]
                # print(f'abp_raw_true: {abp_raw_true}, len: {len(abp_raw_true)}')
                min_f, max_f = abp_f_true.min(), abp_f_true.max()
                # print(f'min_f: {min_f}, max_f: {max_f}')
                min_raw, max_raw = abp_raw_true.min(), abp_raw_true.max()
                # print(f'min_raw: {min_raw}, max_raw: {max_raw}')
                pred_raw = (pred_f - min_f) / (max_f - min_f) * (max_raw - min_raw) + min_raw
                # print(f'pred_raw: {pred_raw}')
                # 计算 ABP_RAW 的 MAE
                pred_raw_sbp = pred_raw[np.isin(all_points, sbp_points)]
                pred_raw_dbp = pred_raw[np.isin(all_points, dbp_points)]
                # print(f'pred_raw_sbp: {pred_raw_sbp}')
                # print(f'pred_raw_dbp: {pred_raw_dbp}')
                mae_raw_sbp += np.abs(pred_raw_sbp.numpy() - abp_raw_true[sbp_points[:len(pred_raw_sbp)]]).mean()
                mae_raw_dbp += np.abs(pred_raw_dbp.numpy() - abp_raw_true[dbp_points[:len(pred_raw_dbp)]]).mean()
                # print(f'mae_raw_sbp: {mae_raw_sbp}')
                # input(f'mae_raw_dbp: {mae_raw_dbp}')
                total_samples += 1

    # 计算平均 MAE
    if total_samples > 0:
        mae_f_sbp /= total_samples
        mae_f_dbp /= total_samples
        mae_raw_sbp /= total_samples
        mae_raw_dbp /= total_samples
    else:
        print("Warning: No valid samples found for evaluation.")

    return {
        'MAE_F_SBP': mae_f_sbp,
        'MAE_F_DBP': mae_f_dbp,
        'MAE_RAW_SBP': mae_raw_sbp,
        'MAE_RAW_DBP': mae_raw_dbp
    }
       
def main():
    train_folder = "tensor_data/train"
    val_folder = "tensor_data/test"
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model_path = 'transformer_ppg_to_abp.pth'
    model = PPGtoBPTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {param_count}")

    train_dataset = BatchDataset(train_folder)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    
    val_dataset = BatchDataset(val_folder)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    model.load_state_dict(torch.load(model_path, map_location=device))
    trained_model = train_model(model, train_loader, val_loader, model_path, epochs=500, lr=1e-6)
    
    torch.save(trained_model.state_dict(), 'transformer_ppg_to_abp.pth')
    h5_folder = "processed_data"
    model.load_state_dict(torch.load(model_path, map_location=device))
    h5_files = [f for f in os.listdir(h5_folder) if f.endswith('.h5')]
    train_files, test_files = train_test_split(h5_files, test_size=0.005, random_state=42)
    results = evaluate(model, test_files, h5_folder, device)
    print("Evaluation Results:")
    print(f"MAE for ABP_F - SBP: {results['MAE_F_SBP']:.4f}, DBP: {results['MAE_F_DBP']:.4f}")
    print(f"MAE for ABP_RAW - SBP: {results['MAE_RAW_SBP']:.4f}, DBP: {results['MAE_RAW_DBP']:.4f}")

if __name__ == "__main__":
    main()
