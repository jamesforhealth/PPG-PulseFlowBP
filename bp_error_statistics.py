import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

##############################################
# 全域參數：ABP 反還原參數（請根據實際資料調整）
##############################################
ABP_OFFSET = 50.0
ABP_SCALE = 100.0

##############################################
# 0) Dataset 定義：讀取 h5 檔資料（個人化，每個 h5 檔代表一位受試者）
##############################################
class BPDataset(Dataset):
    """
    從 .h5 中讀取:
      - ppg: (N,1250)
      - ecg: (N,1250) 若不存在，則以 zeros 取代
      - segsbp, segdbp: (N,)
      - personal_info: (N, M) 若不存在，則以 zeros 填充 (預設 M=5)
      - vascular_properties: (N,3) 若不存在，則以 zeros 填充 (假設 3 維)
         ※ 但根據錯誤訊息，您個人化資料中的 vascular_properties 實際為 2 維，
           所以最終輸出 tensor shape 可能為 (B,2)。
    """
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
            'vascular': self.vascular[idx]          # shape=(?,) 可能實際為 (2,) 依資料而定
        }

##############################################
# 1) Model 定義：ModelPPGECG（此處採用之前的定義）
##############################################
# 以下僅保留 ModelPPGECG 定義，請確保此模型與您訓練使用的版本一致
class ResUNet1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, base_ch=16):
        super().__init__()
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
    def forward(self, x):
        c1 = self.enc_conv1(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        b = self.bottleneck(c4)
        d1 = self.up1(b, c3)
        d2 = self.up2(d1, c2)
        d3 = self.up3(d2, c1)
        out = self.final(d3)
        return out

class MultiHeadSelfAttn1D(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        B,C,L = x.shape
        if C != self.ln.normalized_shape[0]:
            raise ValueError(f"Expected channel {self.ln.normalized_shape[0]}, got {C}")
        x_t = x.transpose(1,2)
        out, _ = self.mha(x_t, x_t, x_t)
        out = self.ln(out)
        out = out.transpose(1,2)
        return out

class ModelPPGECG(nn.Module):
    """
    模型架構：
      - 分別用 ResUNet1D 處理 ppg 與 ecg 信號
      - 各自經 self-attention 及 global average pooling 得到特徵向量 (B, wave_out_ch)
      - 個人資訊經 info_fc 處理成 (B, 32)
      - vascular_properties 經 vasc_fc 處理成 (B, X)；  
        根據您的資料實際情況，這裡 X 可能為 2（若您的資料只有兩個血管參數）
      - 將上述特徵串接後 (B, wave_out_ch*2 + 32 + X)，再經全連接層預測血壓 (2 維: [SBP, DBP])
    """
    def __init__(self, info_dim=4, vascular_dim=2, wave_out_ch=64, d_model=64, n_heads=4):
        super().__init__()
        self.ppg_unet = ResUNet1D(in_ch=1, out_ch=wave_out_ch)
        self.ecg_unet = ResUNet1D(in_ch=1, out_ch=wave_out_ch)
        self.self_attn_ppg = MultiHeadSelfAttn1D(d_model=d_model, n_heads=n_heads)
        self.self_attn_ecg = MultiHeadSelfAttn1D(d_model=d_model, n_heads=n_heads)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        # personal info 處理
        self.info_fc = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU()
        )
        # vascular_properties 處理，注意 vascular_dim 這裡設定為 2
        self.vasc_fc = nn.Sequential(
            nn.Linear(vascular_dim, 32),
            nn.ReLU()
        )
        # 最終全連接層：輸入維度 = wave_out_ch*2 + 32 + 32
        self.final_fc = nn.Sequential(
            nn.Linear(wave_out_ch*2 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, ppg, ecg, personal_info, vascular):
        ppg_feat_map = self.ppg_unet(ppg)      # (B, wave_out_ch, L)
        ecg_feat_map = self.ecg_unet(ecg)        # (B, wave_out_ch, L)
        ppg_feat_map = self.self_attn_ppg(ppg_feat_map)  # (B, wave_out_ch, L)
        ecg_feat_map = self.self_attn_ecg(ecg_feat_map)    # (B, wave_out_ch, L)
        ppg_feat = self.final_pool(ppg_feat_map).squeeze(-1)  # (B, wave_out_ch)
        ecg_feat = self.final_pool(ecg_feat_map).squeeze(-1)  # (B, wave_out_ch)
        info_feat = self.info_fc(personal_info)   # (B,32)
        vasc_feat = self.vasc_fc(vascular)         # (B,32)
        combined = torch.cat([ppg_feat, ecg_feat, info_feat, vasc_feat], dim=1)  # (B, wave_out_ch*2 + 64)
        out = self.final_fc(combined)  # (B,2)
        return out

##############################################
# 3) 預測並彙整受試者結果的函式
##############################################
def predict_for_subject(subject_file, model, device='cuda'):
    """
    讀取單個 h5 檔（代表一位受試者），利用模型預測所有 segment 的血壓，
    並取平均作為該受試者的預測結果，同時計算平均真值。
    另外讀取該受試者的個人資訊與 vascular_properties（取第一筆），並計算 BMI。
    回傳字典。
    """
    ds = BPDataset(str(subject_file))
    dl = DataLoader(ds, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    all_preds = []
    all_labels = []
    for batch in dl:
        ppg = batch['ppg'].to(device)  # (B,1,1250)
        ecg = batch['ecg'].to(device)  # (B,1,1250)
        pi = batch['personal_info'].to(device)  # (B,4)
        vas = batch['vascular'].to(device)        # (B,?)
        with torch.no_grad():
            out = model(ppg, ecg, pi, vas)  # (B,2)
        all_preds.append(out.cpu().numpy())
        all_labels.append(batch['bp_label'].numpy())
    all_preds = np.concatenate(all_preds, axis=0)  # (N,2)
    all_labels = np.concatenate(all_labels, axis=0)  # (N,2)
    # 平均預測與真值（均為 normalized，還原回 mmHg）
    sbp_pred_norm = np.mean(all_preds[:,0])
    dbp_pred_norm = np.mean(all_preds[:,1])
    sbp_true_norm = np.mean(all_labels[:,0])
    dbp_true_norm = np.mean(all_labels[:,1])
    sbp_pred = sbp_pred_norm * ABP_SCALE + ABP_OFFSET
    dbp_pred = dbp_pred_norm * ABP_SCALE + ABP_OFFSET
    sbp_true = sbp_true_norm * ABP_SCALE + ABP_OFFSET
    dbp_true = dbp_true_norm * ABP_SCALE + ABP_OFFSET
    # 從第一筆取個人資訊
    personal_info = ds[0]['personal_info'].numpy()  # (4,)
    age = personal_info[0]
    gender = personal_info[1]
    weight = personal_info[2]
    height = personal_info[3]
    BMI = weight / ((height/100)**2) if height > 0 else np.nan
    result = {
        "subject_id": subject_file.stem,
        "age": age,
        "gender": gender,
        "BMI": BMI,
        "sbp_true": sbp_true,
        "dbp_true": dbp_true,
        "sbp_pred": sbp_pred,
        "dbp_pred": dbp_pred,
        "num_segments": len(ds)
    }
    return result

##############################################
# 4) 統計分析與圖表產生
##############################################
def analyze_and_plot(results_csv="subject_prediction_results.csv"):
    df = pd.read_csv(results_csv)
    df['mae_sbp'] = np.abs(df['sbp_true'] - df['sbp_pred'])
    df['mae_dbp'] = np.abs(df['dbp_true'] - df['dbp_pred'])
    
    # 建立年齡分組
    bins_age = [0, 30, 60, 150]
    labels_age = ['<30', '30-60', '>=60']
    df['age_group'] = pd.cut(df['age'], bins=bins_age, labels=labels_age)
    
    # 建立 BMI 分組
    bins_bmi = [0, 18.5, 25, 30, 100]
    labels_bmi = ['偏瘦','正常','過重','肥胖']
    df['BMI_group'] = pd.cut(df['BMI'], bins=bins_bmi, labels=labels_bmi)
    
    # 血壓分類（以 sbp 為例）
    def classify_bp(sbp):
        if sbp < 120:
            return 'Normal'
        elif sbp < 140:
            return 'Mild Hypertension'
        elif sbp < 160:
            return 'Moderate Hypertension'
        else:
            return 'Severe Hypertension'
    df['BP_group'] = df['sbp_true'].apply(classify_bp)
    
    sns.set(style="whitegrid", font_scale=1.2)
    
    # 1. 箱型圖：按年齡分組
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    sns.boxplot(x='age_group', y='mae_sbp', data=df)
    plt.title("按年齡組 SBP MAE")
    plt.xlabel("年齡分組")
    plt.ylabel("SBP MAE (mmHg)")
    
    plt.subplot(122)
    sns.boxplot(x='age_group', y='mae_dbp', data=df)
    plt.title("按年齡組 DBP MAE")
    plt.xlabel("年齡分組")
    plt.ylabel("DBP MAE (mmHg)")
    plt.tight_layout()
    plt.savefig("mae_by_age_group.png")
    plt.close()
    
    # 2. 箱型圖：按 BMI 分組
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    sns.boxplot(x='BMI_group', y='mae_sbp', data=df)
    plt.title("按 BMI 組 SBP MAE")
    plt.xlabel("BMI 分組")
    plt.ylabel("SBP MAE (mmHg)")
    
    plt.subplot(122)
    sns.boxplot(x='BMI_group', y='mae_dbp', data=df)
    plt.title("按 BMI 組 DBP MAE")
    plt.xlabel("BMI 分組")
    plt.ylabel("DBP MAE (mmHg)")
    plt.tight_layout()
    plt.savefig("mae_by_bmi_group.png")
    plt.close()
    
    # 3. 箱型圖：按血壓分類
    plt.figure(figsize=(12,5))
    order_bp = ['Normal','Mild Hypertension','Moderate Hypertension','Severe Hypertension']
    plt.subplot(121)
    sns.boxplot(x='BP_group', y='mae_sbp', data=df, order=order_bp)
    plt.title("按血壓分類 SBP MAE")
    plt.xlabel("血壓分類")
    plt.ylabel("SBP MAE (mmHg)")
    
    plt.subplot(122)
    sns.boxplot(x='BP_group', y='mae_dbp', data=df, order=order_bp)
    plt.title("按血壓分類 DBP MAE")
    plt.xlabel("血壓分類")
    plt.ylabel("DBP MAE (mmHg)")
    plt.tight_layout()
    plt.savefig("mae_by_bp_group.png")
    plt.close()
    
    # 4. 散點圖：預測誤差與真值之間的關係
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    sns.scatterplot(x='sbp_true', y='mae_sbp', hue='age_group', data=df)
    plt.title("SBP 真值 vs SBP MAE")
    plt.xlabel("SBP 平均真值 (mmHg)")
    plt.ylabel("SBP MAE (mmHg)")
    
    plt.subplot(122)
    sns.scatterplot(x='dbp_true', y='mae_dbp', hue='age_group', data=df)
    plt.title("DBP 真值 vs DBP MAE")
    plt.xlabel("DBP 平均真值 (mmHg)")
    plt.ylabel("DBP MAE (mmHg)")
    plt.tight_layout()
    plt.savefig("scatter_mae_vs_bp.png")
    plt.close()
    
    # 5. ANOVA 檢定
    groups = [group["mae_sbp"].values for name, group in df.groupby("age_group")]
    f_val, p_val = stats.f_oneway(*groups)
    print(f"年齡組 SBP MAE ANOVA 測試: F={f_val:.3f}, p={p_val:.3f}")
    
    df.to_csv("subject_prediction_results.csv", index=False)
    print("聚合結果已保存至 subject_prediction_results.csv")

##############################################
# 4. DataLoader 建立（針對個人化資料夾）
##############################################
def create_subject_dataloaders(subject_folder="personalized_training_data_VitalDB"):
    subject_folder = Path(subject_folder)
    subject_files = list(subject_folder.glob("*.h5"))
    return subject_files

##############################################
# 5. 主程式：針對每個受試者進行預測並統計
##############################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 載入模型與權重（請確保 model_ModelPPGECG.pth 存在）
    from bp_resunet_attention_compare import ModelPPGECG
    # 注意：若您的資料實際 vascular_properties 維度為 2，則請設 vascular_dim=2
    model = ModelPPGECG(info_dim=4, vascular_dim=3, wave_out_ch=32, d_model=32, n_heads=4)
    model_path = Path("model_ModelPPGECG.pth")
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Model weight file {model_path} not found.")
    model.to(device)
    model.eval()
    
    subject_files = create_subject_dataloaders("personalized_training_data_VitalDB")
    results_list = []
    for sf in tqdm(subject_files, desc="Predicting Subjects"):
        res = predict_for_subject(sf, model, device=device)
        results_list.append(res)
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("subject_prediction_results.csv", index=False)
    print("Saved subject-level prediction results to subject_prediction_results.csv")
    
    # 統計分析與圖表產生
    analyze_and_plot("subject_prediction_results.csv")

if __name__ == "__main__":
    main()
