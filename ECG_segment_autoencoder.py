import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

def load_mimic_data_with_split(h5_path, val_ratio=0.2):
    with h5py.File(h5_path, 'r') as f:
        ecg_data = f['ecg'][:]
    # ecg_data.shape = (n_segments, 1250)

    train_data, val_data = train_test_split(
        ecg_data, 
        test_size=val_ratio, 
        random_state=42  # for reproducibility
    )
    return train_data, val_data

#-----------------------------------------------------------------------------
# 1) 縮小模型容量: 將 channel 改為 8->16->32->64 (原本16->32->64->128)
#-----------------------------------------------------------------------------
class ECGAutoencoder(nn.Module):
    def __init__(self, seq_len=1250):
        super(ECGAutoencoder, self).__init__()
        
        # Encoder (1250 -> 625 -> 313 -> 157 -> 79)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),   # 1250 -> 625
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),  # 625 -> 313
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), # 313 -> 157
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), # 157 -> 79
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # Decoder (79 -> 157 -> 313 -> 625 -> 1250)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#-----------------------------------------------------------------------------
# 2) 加載資料 (不變)
#-----------------------------------------------------------------------------
def load_mimic_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ecg_data = f['ecg'][:]  # 假設ecg數據存於'ecg'
    return ecg_data

#-----------------------------------------------------------------------------
# 3) 訓練函式: 加入 Sparse + Denoising 邏輯
#-----------------------------------------------------------------------------
def train_autoencoder(train_data,
                      val_data,
                      device='cuda', 
                      epochs=100, 
                      batch_size=64, 
                      model_path='ecg_autoencoder.pth',
                      use_denoising=False,
                      noise_std=0.1,       # Denoising時的高斯雜訊標準差
                      use_sparse=False,
                      sparse_lambda=1e-3    # 稀疏懲罰係數
                      ):
    """
    train_data, val_data: numpy array, shape (n_segments, 1250)
    use_denoising: 是否啟用Denoising機制
    noise_std: Denoising時在輸入上的高斯雜訊強度
    use_sparse: 是否啟用Sparse機制
    sparse_lambda: 稀疏懲罰的權重(越大越強烈)
    """

    # --- 建立 scaler ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_val   = scaler.transform(val_data)

    # --- 建立 Dataset/DataLoader ---
    x_train_tensor = torch.FloatTensor(scaled_train).unsqueeze(1)  # shape (N,1,1250)
    train_dataset  = TensorDataset(x_train_tensor)
    train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_val_tensor = torch.FloatTensor(scaled_val).unsqueeze(1)
    val_dataset  = TensorDataset(x_val_tensor)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 建立 model/criterion/optimizer ---
    model = ECGAutoencoder().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params}")
    
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Load existing weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_model_state = None

    train_losses = []
    val_losses   = []

    for epoch in tqdm(range(epochs)):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader):
            x = batch[0].to(device)
            optimizer.zero_grad()

            # (A) 如果啟用 Denoising，對 x 加一些雜訊 (只在 train 時)
            if use_denoising:
                noise = torch.randn_like(x) * noise_std
                x_noisy = x + noise
            else:
                x_noisy = x

            # Forward
            output = model(x_noisy)

            # (B) 基本重構損失
            recon_loss = criterion(output, x)

            # (C) 如果啟用 Sparse，針對隱層activation加懲罰
            #     這裡示範 "所有 encoder 最後一層" 的 activations 做 L1
            #     也可在 forward() 直接回傳 encoded, decoded 來自定義
            sparse_loss = 0.0
            if use_sparse:
                # 取出 encoder 最後一層輸出 (model.encoder[-1] 是最後module)
                # 但更嚴謹做法：在 forward() 回傳encoded 以確保拿到正確tensor
                with torch.no_grad():
                    encoded = model.encoder(x_noisy)
                # L1懲罰: sum(|encoded|)
                # encoded shape = (batch_size, 64, 79)，我們把所有elements加起來
                sparse_loss = torch.mean(torch.abs(encoded))  # 取平均也可

            total_loss = recon_loss + sparse_lambda * sparse_loss

            # Backprop
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_val = batch[0].to(device)

                if use_denoising:
                    # 評估時一般不加雜訊 (除非想測試穩健性)
                    output_val = model(x_val)
                else:
                    output_val = model(x_val)

                recon_loss_val = criterion(output_val, x_val)
                
                # Sparse懲罰(驗證時是否要加? 可看需求)
                sparse_loss_val = 0.0
                if use_sparse:
                    encoded_val = model.encoder(x_val)
                    sparse_loss_val = torch.mean(torch.abs(encoded_val))

                val_loss = recon_loss_val + sparse_lambda * sparse_loss_val
                total_val_loss += val_loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # 顯示當前 epoch 訓練和驗證 loss
        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Train Loss: {epoch_train_loss:.6f}  "
              f"Val Loss: {epoch_val_loss:.6f}")

        # ---- 若 val_loss 變更小，就記錄當前最好的模型狀態 ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            print(f"[INFO] Best val loss so far: {best_val_loss:.6f}, saving model.")
            torch.save(best_model_state, model_path)

    # ---- 重新載入最佳模型權重 ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] Final best val loss: {best_val_loss:.6f}, loaded best weights.")
    else:
        print("[WARNING] No best model state found (val_loss never updated?).")

    return model, scaler, train_losses, val_losses


def get_reconstruction_errors(model, data, scaler, device='cuda', batch_size=64):
    """計算所有片段的重構誤差"""
    model.eval()
    scaled_data = scaler.transform(data)
    x_test = torch.FloatTensor(scaled_data).unsqueeze(1)
    dataset = TensorDataset(x_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    reconstruction_errors = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            output = model(x)
            errors = torch.mean((output - x) ** 2, dim=(1,2))
            reconstruction_errors.extend(errors.cpu().numpy())
    
    return np.array(reconstruction_errors)

def plot_results(losses, reconstruction_errors, threshold_percentile=95):
    """可視化訓練過程和重構誤差分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 訓練損失曲線
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE + Regularization')
    
    # 重構誤差分布
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    ax2.hist(reconstruction_errors, bins=50, density=True)
    ax2.axvline(threshold, color='r', linestyle='--', 
                label=f'{threshold_percentile}th percentile')
    ax2.set_title('Reconstruction Error Distribution')
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def filter_segments(reconstruction_errors, segments, threshold_percentile=95):
    """根據重構誤差篩選片段"""
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    good_indices = reconstruction_errors <= threshold
    filtered_segments = segments[good_indices]
    
    print(f"原始片段數: {len(segments)}")
    print(f"篩選後片段數: {len(filtered_segments)}")
    print(f"過濾掉 {len(segments) - len(filtered_segments)} 個異常片段")
    
    return filtered_segments, good_indices

def main():
    # 1. 加載 MIMIC 數據並分割
    train_data, val_data = load_mimic_data_with_split(
        "training_data_1250_MIMIC_test/training_1.h5", 
        val_ratio=0.2
    )
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 訓練自編碼器: 可切換是否用Denoising / Sparse
    model, scaler, train_losses, val_losses = train_autoencoder(
        train_data, 
        val_data,
        device=device, 
        epochs=50, 
        batch_size=64, 
        model_path="ecg_autoencoder_test.pth",
        use_denoising=True,    # 啟用Denoising
        noise_std=0.1,        # 雜訊 std
        use_sparse=True,       # 啟用Sparse
        sparse_lambda=1e-1     # 稀疏懲罰權重
    )

    # 3. 繪製 train/val 損失曲線
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE + Regularization")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.show()

    # (範例) 若想看測試集或其他資料(如VitalDB)，可載入並測試重構誤差
    # vitaldb_data = load_mimic_data("training_data_VitalDB/training_1.h5")
    # reconstruction_errors = get_reconstruction_errors(model, vitaldb_data, scaler, device)
    # plot_results(val_losses, reconstruction_errors)
    # filtered_segments, good_indices = filter_segments(reconstruction_errors, vitaldb_data)

    return model, scaler

if __name__ == "__main__":
    model, scaler = main()
