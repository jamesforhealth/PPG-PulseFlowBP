##################################################
# multi_signal_fusion_train.py
# PPG (raw, 1st diff, 2nd diff) + ECG => 4通道
# 另有6維特徵(個人4 + 血管2) => MLP嵌入 => 與CNN+LSTM融合 => 預測SBP,DBP
##################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import os
from tqdm import tqdm
############################
# (A) Dataset
############################
class VitalSignDataset(Dataset):
    """
    從單一 .h5 檔案讀取資料:
      ppg: shape=(N,1250)
      ecg: shape=(N,1250)
      annotations: shape=(N,1250,4)  (這邊可能是否用?)
      personal_info: shape=(N,4)
      vascular_properties: shape=(N,2)
      segsbp, segdbp => label=(N,2)

    * 在 __getitem__() 中：
      - 計算 PPG 的一階、二階差分
      - 分別做 0~1 正規化 (對該 segment)
      - 同時對 ECG 做 0~1 正規化
      - 合成 wave: (4,1250)
      - 其餘 6 維 (pers(4)+vasc(2)) => one tensor
      - label => (2,)
    """
    def __init__(self, h5_file):
        super().__init__()
        self.h5 = h5py.File(h5_file, 'r')
        self.ppg = self.h5['ppg']               # (N,1250)
        self.ecg = self.h5['ecg']               # (N,1250)
        self.annotations = self.h5['annotations']  # (N,1250,4) (如需要可用)
        self.personal_info = self.h5['personal_info']  # (N,4)
        self.vascular = self.h5['vascular_properties'] # (N,2)
        self.labels = np.stack([self.h5['segsbp'], self.h5['segdbp']], axis=1)  # (N,2)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        # 讀取 numpy array
        ppg_data  = self.ppg[idx]   # (1250,)
        ecg_data  = self.ecg[idx]   # (1250,)
        pers_info = self.personal_info[idx]   # (4,)
        vasc_info = self.vascular[idx]        # (2,)
        label_data= self.labels[idx]          # (2,)

        # ----------- 計算 PPG 一階/二階差分 -----------
        ppg_1st_diff = np.zeros_like(ppg_data)
        # 中心差分 (1~(end-1))
        ppg_1st_diff[1:-1] = (ppg_data[2:] - ppg_data[:-2]) / 2
        # 邊界
        ppg_1st_diff[0] = ppg_data[1] - ppg_data[0]
        ppg_1st_diff[-1] = ppg_data[-1] - ppg_data[-2]

        ppg_2nd_diff = np.zeros_like(ppg_data)
        # 中心差分 for 2nd
        # ppg_2nd_diff[2:-2] = (ppg_data[4:] - 2*ppg_data[2:-2] + ppg_data[:-4]) / (2^2) => or 4
        # 這裡給個簡化: denominator=4
        ppg_2nd_diff[2:-2] = (ppg_data[4:] - 2*ppg_data[2:-2] + ppg_data[:-4]) / 4
        # 邊界用附近值
        ppg_2nd_diff[0]   = ppg_2nd_diff[2]
        ppg_2nd_diff[1]   = ppg_2nd_diff[2]
        ppg_2nd_diff[-2]  = ppg_2nd_diff[-3]
        ppg_2nd_diff[-1]  = ppg_2nd_diff[-3]

        # ----------- 0~1 Normalization -----------
        # 針對 ppg_data, ppg_1st_diff, ppg_2nd_diff, ecg_data 各自獨立做 min-max => 0~1
        def minmax_norm(sig):
            mn = sig.min()
            mx = sig.max()
            if mx == mn:
                return np.zeros_like(sig)  # or all-zero if no variation
            return (sig - mn)/(mx - mn)

        ppg_raw_norm = minmax_norm(ppg_data)
        ppg_1st_norm = minmax_norm(ppg_1st_diff)
        ppg_2nd_norm = minmax_norm(ppg_2nd_diff)
        ecg_norm     = minmax_norm(ecg_data)

        # ----------- 合併成 wave: shape=(4,1250) -----------
        # channel順序: [ppg_raw, ppg_1st, ppg_2nd, ecg]
        wave_4ch = np.stack([ppg_raw_norm, ppg_1st_norm, ppg_2nd_norm, ecg_norm], axis=0)  # (4,1250)

        # ----------- 特徵6維 => (pers_info(4) + vasc_info(2)) -----------
        extra_feat = np.concatenate([pers_info, vasc_info], axis=0)  # => shape(6,)

        # ----------- torch 化 -----------
        wave_t  = torch.from_numpy(wave_4ch).float()    # (4,1250)
        extra_t = torch.from_numpy(extra_feat).float()  # (6,)
        label_t = torch.from_numpy(label_data).float()  # (2,)

        return wave_t, extra_t, label_t


############################
# (B) Model: CNN+LSTM on wave(4ch), MLP on extra(6dim)
############################
class MultiSignalFusionBPEstimator(nn.Module):
    """
    流程:
     1) wave(4,1250) => CNN => pool => shape=(batch,channel',time')
     2) => permute => LSTM => (batch,hidden_dim)
     3) extra(6) => MLP => embed_dim
     4) concat => final => (2)
    """
    def __init__(self, hidden_dim=64, extra_emb_dim=32):
        super().__init__()
        self.wave_channels = 4  # ppg_raw, ppg_1st, ppg_2nd, ecg
        # CNN
        self.conv1 = nn.Conv1d(self.wave_channels, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)  # 1250->625

        # LSTM
        self.lstm  = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)

        # Extra(6) => MLP => embed_dim(32)
        self.extra_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, extra_emb_dim),
            nn.ReLU()
        )
        # final => hidden_dim + extra_emb_dim => 2
        self.fc_final = nn.Linear(hidden_dim + extra_emb_dim, 2)

    def forward(self, wave, extra):
        """
        wave: (batch,4,1250)
        extra: (batch,6)
        """
        # 1) CNN
        x = F.relu(self.bn1(self.conv1(wave)))  # =>(B,32,1250)
        x = F.relu(self.bn2(self.conv2(x)))     # =>(B,64,1250)
        x = self.pool(x)                        # =>(B,64,625)

        # 2) LSTM => (B,625,64)
        x = x.permute(0,2,1)
        out_lstm, (hn, cn) = self.lstm(x)
        # 取最後 time step => (B, hidden_dim)
        feat_seq = out_lstm[:,-1,:]

        # 3) Extra => MLP
        feat_extra = self.extra_mlp(extra)  # =>(B, extra_emb_dim)

        # 4) concat => final => (B,2)
        comb = torch.cat([feat_seq, feat_extra], dim=1)
        out = self.fc_final(comb)
        return out


############################
# (B) ResBlock + CBAM modules
############################

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # Spatial attention (1D)
        self.conv_spatial = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        b,c,t = x.shape
        # channel attn
        avgout = self.fc(self.avg_pool(x).view(b,c))
        maxout = self.fc(self.max_pool(x).view(b,c))
        scale = torch.sigmoid(avgout + maxout).unsqueeze(-1)  # (b,c,1)
        x = x * scale

        # spatial attn
        avg_spatial = torch.mean(x, dim=1, keepdim=True) # (b,1,t)
        max_spatial,_= torch.max(x, dim=1, keepdim=True) # (b,1,t)
        cat_spatial = torch.cat([avg_spatial, max_spatial], dim=1) # (b,2,t)
        spatial_scale = torch.sigmoid(self.conv_spatial(cat_spatial)) # (b,1,t)
        x = x * spatial_scale
        return x

class ResBlock(nn.Module):
    """
    Simple ResBlock with optional stride + CBAM
    in_ch -> out_ch
    """
    def __init__(self, in_ch, out_ch, stride=1, use_cbam=False):
        super().__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.downsample = None
        if stride!=1 or in_ch!=out_ch:
            self.downsample= nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        if use_cbam:
            self.cbam = CBAM(out_ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        if self.use_cbam:
            out = self.cbam(out)
        return out

############################
# (C) rU-Net + BiGRU + Demographic
############################
class RUResAttnBPEstimator(nn.Module):
    """
    2層 encoder + 2層 decoder
    - enc1: in_ch=4 => out_ch=32
      + demographic => cat => (32+dem_fc1=32) => 64? => feed enc1 -> out: 32 => cat dem => 64
    - enc2: in_ch=64 => out_ch=64, stride=2
      + use_cbam=True at this deeper level
      => out: 64 => cat dem => 96 (just as example, or keep 64)
    - dec2: upsample => in_ch=96 => out_ch=64
    - dec1: upsample => in_ch=64 => out_ch=32 => final 1x1 => (32)

    - BiGRU(32) => final(2)
    """
    def __init__(self, base_ch=32, gru_hidden=64):
        super().__init__()
        self.base_ch= base_ch

        # dem fcs
        self.dem_fc1 = nn.Linear(6, base_ch)
        self.dem_fc2 = nn.Linear(6, base_ch)

        # enc1
        self.enc1_rsb = ResBlock(in_ch=4, out_ch=base_ch, stride=1, use_cbam=False)

        # enc2
        # => after enc1 out => base_ch => + dem => base_ch => total = 2*base_ch
        # feed to ResBlock => output ch= base_ch*2? 這裡簡化 => out_ch= base_ch (or base_ch*2).
        # 為避免 mismatch, let's do out_ch=64 => i.e. base_ch=32 => out_ch=64
        self.enc2_rsb = ResBlock(in_ch=(base_ch*2), out_ch=base_ch*2, stride=2, use_cbam=True)

        # decode2
        # after enc2 => shape=(b, base_ch*2, t/2?). plus dem =>  (base_ch*2 + base_ch= base_ch*3)? let's do simpler approach:
        self.up2= nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2_rsb= ResBlock(in_ch=base_ch*2, out_ch=base_ch, stride=1, use_cbam=False)

        # decode1
        # up => shape=(b, base_ch, t) => final conv => base_ch
        self.up1= nn.Upsample(scale_factor=1, mode='nearest')  # no change?
        self.dec1_rsb= ResBlock(in_ch=base_ch, out_ch=base_ch, stride=1, use_cbam=False)

        self.out_conv= nn.Conv1d(base_ch, base_ch, kernel_size=1)

        # BiGRU => input_size= base_ch
        self.gru= nn.GRU(input_size=base_ch, hidden_size=gru_hidden, batch_first=True, bidirectional=True)
        self.fc_out= nn.Linear(gru_hidden*2, 2)

    def forward(self, wave, extra):
        b,c,t= wave.shape
        # enc1
        # demographic => fc1 => shape=(b,base_ch) => tile => shape=(b,base_ch,t)
        dem1= self.dem_fc1(extra).unsqueeze(-1).repeat(1,1,t) # =>(b,base_ch,t)
        x1= self.enc1_rsb(wave) # =>(b, base_ch, t)
        x1= torch.cat([x1, dem1], dim=1) # =>(b, base_ch+base_ch=2*base_ch, t)= (b,64,t)

        # enc2
        x2= self.enc2_rsb(x1) # =>(b, base_ch*2=64, t/2)
        # no further dem cat here to simplify; or you can cat dem2 => shape mismatch => handle carefully

        # decode2
        up2= self.up2(x2) # => scale_factor=2 => back to t
        x_dec2= self.dec2_rsb(up2) # =>(b, base_ch=32, t)

        # decode1
        up1= self.up1(x_dec2) # scale_factor=1 => no change
        x_dec1= self.dec1_rsb(up1) # =>(b, base_ch=32, t)

        x_out= self.out_conv(x_dec1) # =>(b, base_ch, t) = (b,32,t)

        # BiGRU
        x_out= x_out.permute(0,2,1) # =>(b,t,32)
        out_gru, _= self.gru(x_out) # =>(b,t, hidden*2)
        feat_seq= out_gru[:,-1,:]   # =>(b, hidden*2)
        out2ch= self.fc_out(feat_seq) # =>(b,2)
        return out2ch

############################
# (D) Evaluate & Train
############################
def evaluate_model_mae(model, dataloader, device='cpu'):
    """
    分別評估 SBP 和 DBP 的 MAE
    """
    model.eval()
    total_mae_sbp = 0.0
    total_mae_dbp = 0.0
    total_count = 0
    
    with torch.no_grad():
        for wave, extra, labels in dataloader:
            wave = wave.to(device)
            extra = extra.to(device)
            labels = labels.to(device)
            
            preds = model(wave, extra)
            mae_sbp = torch.sum(torch.abs(preds[:,0] - labels[:,0]))
            mae_dbp = torch.sum(torch.abs(preds[:,1] - labels[:,1]))
            
            batch_size = wave.size(0)
            total_mae_sbp += mae_sbp.item()
            total_mae_dbp += mae_dbp.item()
            total_count += batch_size
    
    avg_mae_sbp = total_mae_sbp / total_count
    avg_mae_dbp = total_mae_dbp / total_count
    
    return avg_mae_sbp, avg_mae_dbp

def train_model(model, dataloaders, optimizer, num_epochs=50, device='cpu'):
    criterion = nn.L1Loss()
    
    # 初始評估
    init_val_mae_sbp, init_val_mae_dbp = evaluate_model_mae(model, dataloaders['val'], device=device)
    print(f"[Initial] val MAE - SBP: {init_val_mae_sbp:.4f}, DBP: {init_val_mae_dbp:.4f}")
    
    init_test_mae_sbp, init_test_mae_dbp = evaluate_model_mae(model, dataloaders['test'], device=device)
    print(f"[Initial] test MAE - SBP: {init_test_mae_sbp:.4f}, DBP: {init_test_mae_dbp:.4f}")

    best_wts = model.state_dict()
    best_mae = (init_val_mae_sbp + init_val_mae_dbp) / 2  # 或者您可以選擇其他方式來決定最佳模型

    for epoch in tqdm(range(1, num_epochs+1)):
        print(f"Epoch {epoch}/{num_epochs}")
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total_samples = 0
            
            for wave, extra, labels in tqdm(dataloaders[phase]):
                wave = wave.to(device)
                extra = extra.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    preds = model(wave, extra)
                    loss = criterion(preds, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * wave.size(0)
                total_samples += wave.size(0)

            epoch_loss = running_loss / total_samples
            
            if phase == 'val':
                val_mae_sbp, val_mae_dbp = evaluate_model_mae(model, dataloaders['val'], device=device)
                print(f"{phase} MAE - SBP: {val_mae_sbp:.4f}, DBP: {val_mae_dbp:.4f}")
                
                # 使用平均 MAE 來決定是否保存模型
                current_mae = (val_mae_sbp + val_mae_dbp) / 2
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_wts = model.state_dict()
            else:
                print(f"{phase} Loss: {epoch_loss:.4f}")
                
        print("--------------------------------------------------")

    print(f"Training done, best val MAE: {best_mae:.4f}")
    test_mae_sbp, test_mae_dbp = evaluate_model_mae(model, dataloaders['test'], device=device)
    print(f"Test MAE - SBP: {test_mae_sbp:.4f}, DBP: {test_mae_dbp:.4f}")
    
    model.load_state_dict(best_wts)
    return model


############################
# (D) main
############################

def get_age_range_index(age):
    if age < 30:
        return 0
    elif age < 60:
        return 1
    else:
        return 2

def get_gender_index(gender):
    # 假設 0=male, 1=female，或反之
    if gender == 70:
        return 0
    else:
        return 1

def get_subset_index(age, gender):
    # subset_index = gender_index*3 + age_index
    # 0: male<30, 1: male30-60, 2: male>=60
    # 3: female<30,4: female30-60, 5: female>=60
    ar = get_age_range_index(age)
    gr = get_gender_index(gender)
    return gr * 3 + ar

def split_dataset_by_6groups(dataset):
    """
    將 dataset (wave, extra, label) 依照 (gender, age) 分為 6 個 list，
    分別回傳 6 個 SubsetDataset。
    """
    from torch.utils.data import Dataset, Subset
    
    groups_data = [[] for _ in range(6)]  # list of lists

    for idx in tqdm(range(len(dataset))):
        wave, extra, label = dataset[idx]
        # extra 形狀 (6,) => personal(4)+vascular(2)
        # 這裡假設 extra[:2] => [age, gender] (若您實際順序不同，請對應修正)
        # 若 personal_info = (age, gender, weight, height),
        #   vascular = (ptt, pat),
        #   則 extra[0]=age, extra[1]=gender, extra[2]=weight, extra[3]=height, extra[4],extra[5]...
        #   請依自己實際實裝對應
        age_val = extra[0].item()
        gender_val = extra[1].item()
        # print(f"age_val: {age_val}, gender_val: {gender_val}")
        subset_idx = get_subset_index(age_val, gender_val)
        # print(f"subset_idx: {subset_idx}")
        groups_data[subset_idx].append(idx)
    
    # 轉成 Subset
    subsets = []
    for i in range(6):
        if len(groups_data[i])==0:
            print(f"Group {i} is empty!")
        subset_i = Subset(dataset, groups_data[i])
        subsets.append(subset_i)
    return subsets

def freeze_layers_for_subgroup_finetune(model, freeze_cnn=True, freeze_lstm=True):
    """
    根據參數，將 model 中 CNN 及 LSTM 的參數 requires_grad=False，
    只留 extra_mlp + fc_final 可以更新。
    
    您可自行調整要 freeze 哪些層：例如只 freeze conv1, conv2, 
    但保留 LSTM 也 fine-tune；或反之。
    """
    # 1) CNN 區塊
    if freeze_cnn:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        for param in model.bn2.parameters():
            param.requires_grad = False
    
    # 2) LSTM 區塊
    # if freeze_lstm:
    #     for param in model.lstm.parameters():
    #         param.requires_grad = False

    # 3) Extra MLP 與 fc_final 預設不 freeze => 仍可更新

    # optional: 您也可對 self.pool / self.extra_mlp 一併做處理
    # 例如 freeze extra_mlp 的前一層，但保留最後一層 fine-tune...等等


def fine_tune_on_subgroup(model, dataloaders, device='cpu', 
                          freeze_cnn=True, freeze_lstm=True,
                          lr=1e-4, weight_decay=1e-4, num_epochs=10):
    """
    假設 model 已載入 general model 的 state_dict
    然後執行 partial fine-tune on subset data
    """
    # 1) 凍結層
    freeze_layers_for_subgroup_finetune(model, freeze_cnn, freeze_lstm)

    # 2) 只對 requires_grad=True 的參數做 optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
    # 總參數量/可訓練參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in params_to_update)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # 3) 微調
    model= train_model(model, dataloaders, optimizer, num_epochs=num_epochs, device=device)

    return model


def main():
    data_dir= Path('training_data_VitalDB_quality')
    train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
    val_file= data_dir/'validation.h5'
    test_file= data_dir/'test.h5'

    # 1) 讀取 dataset => split by 6 groups
    train_dss=[]
    for tf in train_files:
        if tf.exists():
            train_dss.append(VitalSignDataset(str(tf)))
    full_train_dataset= ConcatDataset(train_dss)
    val_dataset= VitalSignDataset(str(val_file))
    test_dataset= VitalSignDataset(str(test_file))

    train_subsets = split_dataset_by_6groups(full_train_dataset)
    val_subsets   = split_dataset_by_6groups(val_dataset)
    test_subsets  = split_dataset_by_6groups(test_dataset)

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) 先載入 "general model" => e.g. bp_estimator_multisig_fusion.pth
    general_model_path = "bp_estimator_multisig_fusion.pth"
    print(f"Loading general model from {general_model_path} ...")

    for i in range(6):
        print(f"\n===== Fine-tuning subset {i} =====")

        # (a) dataloader
        train_loader_i= DataLoader(train_subsets[i], batch_size=64, shuffle=True, drop_last=True)
        val_loader_i  = DataLoader(val_subsets[i],   batch_size=64, shuffle=False, drop_last=False)
        test_loader_i = DataLoader(test_subsets[i],  batch_size=64, shuffle=False, drop_last=False)
        dataloaders_i = {'train': train_loader_i, 'val': val_loader_i, 'test': test_loader_i}

        # (b) 初始化 model & load general weights
        model_i = MultiSignalFusionBPEstimator(hidden_dim=64, extra_emb_dim=32).to(device)
        model_i.load_state_dict(torch.load(general_model_path))

        # (c) 執行部分層微調
        #    例如 freeze CNN+LSTM，只 fine-tune extra_mlp, fc_final
        model_i = fine_tune_on_subgroup(
            model_i, 
            dataloaders_i,
            device=device,
            freeze_cnn=True,   # freeze CNN
            freeze_lstm=True,  # freeze LSTM
            lr=1e-5,           # 調小一點學習率
            weight_decay=1e-4,
            num_epochs=100
        )

        # (d) 儲存
        out_ckpt = f"bp_estimator_multisig_fusion_sub{i}.pth"
        torch.save(model_i.state_dict(), out_ckpt)
        print(f"Subset {i} fine-tuned model saved to {out_ckpt}")

# ... existing code ...

def visualize_model():
    """
    使用torchviz繪製模型架構圖
    """
    from torchviz import make_dot
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 創建RUResAttnBPEstimator模型實例並移動到指定設備
    model = RUResAttnBPEstimator(base_ch=32, gru_hidden=64).to(device)
    #參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # 創建假數據並移動到相同設備
    batch_size = 1
    wave = torch.randn(batch_size, 4, 1250).to(device)  # (B,4,1250)
    extra = torch.randn(batch_size, 6).to(device)       # (B,6)
    
    # 打印設備信息以確認
    print(f"模型所在設備: {next(model.parameters()).device}")
    print(f"wave所在設備: {wave.device}")
    print(f"extra所在設備: {extra.device}")
    
    # 前向傳播
    out = model(wave, extra)
    
    # 創建視覺化圖
    dot = make_dot(out, params=dict(model.named_parameters()))
    
    # 設置圖片格式
    dot.attr(rankdir='TB')  # 從上到下的布局
    dot.attr('node', shape='box')
    
    # 保存圖片
    dot.render("RUResAttnBPEstimator_architecture", format="png", cleanup=True)
    print("RUResAttnBPEstimator模型架構圖已保存為 'RUResAttnBPEstimator_architecture.png'")


if __name__=='__main__':
    # visualize_model()
    # input("Press Enter to continue...")
    # main()
    data_dir= Path("training_data_VitalDB_quality")
    train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
    val_file= data_dir/'validation.h5'
    test_file= data_dir/'test.h5'

    # 讀 Dataset
    train_dss=[]
    for tf in train_files:
        if tf.exists():
            train_dss.append(VitalSignDataset(str(tf)))
    train_dataset= ConcatDataset(train_dss)
    val_dataset= VitalSignDataset(str(val_file))
    test_dataset= VitalSignDataset(str(test_file))

    train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader  = DataLoader(val_dataset,   batch_size=32, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset,  batch_size=32, shuffle=False, drop_last=False)
    dataloaders= {'train': train_loader, 'val': val_loader, 'test': test_loader}

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device=", device)

    # 建立 rU-Net + 2-lv + CBAM + BiGRU + demographic
    model= RUResAttnBPEstimator(base_ch=32, gru_hidden=64).to(device)
    if os.path.exists("rUNet_cbam_biGRU_demo_fixed.pth"):
        model.load_state_dict(torch.load("rUNet_cbam_biGRU_demo_fixed.pth"))
        print("Model loaded from rUNet_cbam_biGRU_demo_fixed.pth")
    optimizer= optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model= train_model(model, dataloaders, optimizer, num_epochs=50, device=device)

    torch.save(model.state_dict(), "rUNet_cbam_biGRU_demo_fixed.pth")
    print("Done.")

    # data_dir= Path('training_data_VitalDB_quality')
    # train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
    # val_file= data_dir/'validation.h5'
    # test_file= data_dir/'test.h5'

    # train_dss=[]
    # for tf in train_files:
    #     if tf.exists():
    #         train_dss.append(VitalSignDataset(str(tf)))
    # train_dataset= ConcatDataset(train_dss)
    # val_dataset= VitalSignDataset(str(val_file))
    # test_dataset= VitalSignDataset(str(test_file))

    # train_loader= DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    # val_loader  = DataLoader(val_dataset,   batch_size=64, shuffle=False, drop_last=False)
    # test_loader = DataLoader(test_dataset,  batch_size=64, shuffle=False, drop_last=False)
    # dataloaders= {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # # 初始化模型 (CNN+LSTM on wave(4ch), MLP on extra(6d))
    # model= MultiSignalFusionBPEstimator(
    #     hidden_dim=64,     # LSTM hidden
    #     extra_emb_dim=32   # MLP embed dimension for extra 6 dims
    # ).to(device)

    # # 如果已有舊權重，可選擇載入
    # ckpt_path = "bp_estimator_multisig_fusion.pth"
    # if os.path.exists(ckpt_path):
    #     model.load_state_dict(torch.load(ckpt_path))
    #     print(f"Model loaded from {ckpt_path}")

    # total_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params}")

    # optimizer= optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # model= train_model(model, dataloaders, optimizer, num_epochs=5, device=device)

    # torch.save(model.state_dict(), ckpt_path)
    # print("[Done] Model saved.")
