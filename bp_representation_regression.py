import os
import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import argparse

#############################################
# (A) 測試數據集：預處理好的 H5 檔
#############################################
class PreprocessedVitalDataset(Dataset):
    """
    從預處理好的 H5 檔中讀取數據，檔案中需包含：
      - "ppg": (N, segment_length) 例如 1024 點的 PPG segment
      - "age", "gender", "weight", "height", "ptt", "pat", "rr_interval": (N,)
      - "bp": (N,2) 血壓標籤 (SBP, DBP)
    """
    def __init__(self, h5_file: str):
        super().__init__()
        self.h5 = h5py.File(h5_file, 'r')
        self.ppg = self.h5['ppg'][:]           # (N, seg_length)
        self.age = self.h5['age'][:]           
        self.gender = self.h5['gender'][:]     
        self.weight = self.h5['weight'][:]     
        self.height = self.h5['height'][:]     
        self.ptt = self.h5['ptt'][:]           
        self.pat = self.h5['pat'][:]           
        self.rr_interval = self.h5['rr_interval'][:]  
        self.bp = self.h5['bp'][:]             # (N,2)
    def __len__(self) -> int:
        return self.ppg.shape[0]
    def __getitem__(self, idx: int):
        sample = {
            'ppg': self.ppg[idx],  # 1D array, 長度為 segment_length (例如1024)
            'age': self.age[idx],
            'gender': self.gender[idx],
            'weight': self.weight[idx],
            'height': self.height[idx],
            'ptt': self.ptt[idx],
            'pat': self.pat[idx],
            'rr_interval': self.rr_interval[idx],
            'bp': self.bp[idx]     # (2,)
        }
        return sample

#############################################
# (B) 表示學習模型 API 接口
#############################################
def load_pretrained_model(model_type: str, model_path: str, device: torch.device):
    """
    根據 model_type 加載對應的預訓練模型
    要求該模型提供 encode() 方法，輸入張量形狀 (B,1,L)，返回 (B, embedding_dim, T)
    這裡以 VQ-VAE 為例；若 model_type 為 "vae"，則可替換為相應模型
    """
    if model_type.lower() == "vqvae":
        # 請根據實際路徑調整
        from your_vqvae_module import PPG_VQVAE  
        model = PPG_VQVAE(
            input_length=1024,
            input_channels=1,
            embedding_dim=64,
            num_embeddings=128,
            commitment_cost=0.5,
            base_channels=16,
            use_condition=False
        )
    elif model_type.lower() == "vae":
        from your_vae_module import PPG_VAE  
        model = PPG_VAE(input_length=1024, input_channels=1, latent_dim=64, base_channels=16)
    else:
        raise ValueError("未知的模型類型")
    
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def get_latent_vector(ppg_segment: np.ndarray, pretrained_model, device: torch.device) -> np.ndarray:
    """
    對單個 PPG segment (numpy array, 長度 L) 進行推理，返回 latent vector (latent_dim,)
    這裡我們將模型的 encode() 輸出 (B, embedding_dim, T) 沿 T 取平均
    """
    # 轉 tensor，形狀 (1,1,L)
    ppg_tensor = torch.tensor(ppg_segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = pretrained_model.encode(ppg_tensor)  # (1, embedding_dim, T)
    latent_mean = latent.mean(dim=2)  # (1, embedding_dim)
    return latent_mean.squeeze(0).cpu().numpy()  # (embedding_dim,)

#############################################
# (C) 建構回歸數據
#############################################
def build_regression_data(h5_file: str, pretrained_model, device: torch.device):
    """
    讀取預處理後的數據，對每個樣本使用預訓練模型得到 latent 表示，
    將 latent 向量與其他個人特徵串接形成回歸輸入 X，標籤 y 為 bp (2,)
    """
    dataset = PreprocessedVitalDataset(h5_file)
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        latent = get_latent_vector(sample['ppg'], pretrained_model, device)  # (latent_dim,)
        # 其他特徵順序： age, gender, weight, height, ptt, pat, rr_interval  (共7 維)
        additional_feats = np.array([
            sample['age'], sample['gender'], sample['weight'],
            sample['height'], sample['ptt'], sample['pat'], sample['rr_interval']
        ])
        combined = np.concatenate([additional_feats, latent])  # (7+latent_dim,)
        X_list.append(combined)
        y_list.append(sample['bp'])  # (2,)
        if (i + 1) % 10000 == 0:
            print(f"處理了 {i+1} 個樣本...")
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

#############################################
# (D) 回歸模型：傳統 ML 與 DNN
#############################################
def train_random_forest_regressor(X_train, y_train, X_val, y_val):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"[RandomForest] Val MAE: {mae:.4f}")
    return rf, mae

def train_svr_regressor(X_train, y_train, X_val, y_val):
    svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
    svr.fit(X_train, y_train)
    preds = svr.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"[SVR] Val MAE: {mae:.4f}")
    return svr, mae

class DNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=2):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_dnn_regressor(X_train, y_train, X_val, y_val, input_dim, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNRegressor(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = criterion(val_preds, y_val_tensor).item()
        if (epoch+1) % 10 == 0:
            print(f"DNN Epoch {epoch+1}/{epochs}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
    model.eval()
    with torch.no_grad():
        final_preds = model(X_val_tensor).cpu().numpy()
    mae = mean_absolute_error(y_val, final_preds)
    print(f"[DNN] Val MAE: {mae:.4f}")
    return model, mae

#############################################
# (E) 主程式：API接口與回歸測試
#############################################
def main_regression(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 載入預訓練表示模型
    pretrained_model = load_pretrained_model(args.model_type, args.model_path, device)
    # 生成回歸數據
    print("從測試集生成回歸數據...")
    X, y = build_regression_data(args.data_file, pretrained_model, device)
    print(f"Regression data: X shape {X.shape}, y shape {y.shape}")
    # 切分訓練與驗證數據
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    # 輸入特徵維度 = 7 + latent_dim (此處 latent_dim 預設為 64)
    input_dim = 7 + (X_train.shape[1] - 7)
    print("\n=== RandomForest Regression ===")
    rf_model, rf_mae = train_random_forest_regressor(X_train, y_train, X_val, y_val)
    print("\n=== SVR Regression ===")
    svr_model, svr_mae = train_svr_regressor(X_train, y_train, X_val, y_val)
    print("\n=== DNN Regression ===")
    dnn_model, dnn_mae = train_dnn_regressor(X_train, y_train, X_val, y_val, input_dim, epochs=100, lr=1e-3)
    
    # 畫圖比較 MAE
    models = ['RandomForest', 'SVR', 'DNN']
    maes = [rf_mae, svr_mae, dnn_mae]
    plt.figure(figsize=(8,6))
    plt.bar(models, maes)
    plt.title("Regression Model MAE Comparison")
    plt.ylabel("MAE")
    plt.show()
    
    return {"rf_mae": rf_mae, "svr_mae": svr_mae, "dnn_mae": dnn_mae}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression test with pretrained PPG representation")
    parser.add_argument("--model_type", type=str, default="vqvae", help="預訓練模型類型：'vqvae' 或 'vae'")
    parser.add_argument("--model_path", type=str, required=True, help="預訓練模型權重檔案路徑")
    parser.add_argument("--data_file", type=str, required=True, help="預處理後數據檔 (h5 格式)")
    args = parser.parse_args()
    
    results = main_regression(args)
    print("Regression results:", results)
