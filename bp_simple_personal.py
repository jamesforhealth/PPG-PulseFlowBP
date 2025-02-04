import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger= logging.getLogger(__name__)

# ======================================
# (A) Dataset
# ======================================
class SingleSubjectDataset(Dataset):
    """
    單一受試者 .h5:
      ppg, ecg => shape=(N,1250)
      personal_info => (N,4)
      vascular_properties => (N,2)
      segsbp, segdbp => label(2,)

    回傳 (wave, extra, label):
      wave: (2,1250)
      extra: (6,)
      label: (2,) => (sbp,dbp)
    """
    def __init__(self, h5_path):
        super().__init__()
        self.f= h5py.File(h5_path, 'r')
        self.ppg= self.f['ppg'][:]       # (N,1250)
        self.ecg= self.f['ecg'][:]
        self.sbp= self.f['segsbp'][:]
        self.dbp= self.f['segdbp'][:]
        self.personal= self.f['personal_info'][:]
        self.vasc= self.f['vascular_properties'][:]
        self.N= len(self.ppg)

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        ppg_= self.ppg[idx]
        ecg_= self.ecg[idx]
        wave_2ch= np.stack([ppg_, ecg_], axis=0)  # (2,1250)

        pers_= self.personal[idx]   # (4,)
        vasc_= self.vasc[idx]       # (2,)
        extra_6= np.concatenate([pers_, vasc_], axis=0)

        sbp_= self.sbp[idx]
        dbp_= self.dbp[idx]
        label_= np.array([sbp_, dbp_], dtype=np.float32)

        wave_t= torch.from_numpy(wave_2ch).float()
        extra_t= torch.from_numpy(extra_6).float()
        label_t= torch.from_numpy(label_).float()
        return wave_t, extra_t, label_t

# ======================================
# (B) Attention modules
# ======================================
class CBAM1D(nn.Module):
    """簡易 channel+spatial attention (CBAM) for 1D."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channels= channels
        self.reduction= reduction
        # channel
        self.avg_pool= nn.AdaptiveAvgPool1d(1)
        self.max_pool= nn.AdaptiveMaxPool1d(1)
        self.mlp= nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False)
        )
        # spatial
        self.conv_spatial= nn.Conv1d(2,1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        # x:(B,C,T)
        b,c,t= x.shape
        # channel attn
        avgout= self.mlp(self.avg_pool(x).view(b,c))
        maxout= self.mlp(self.max_pool(x).view(b,c))
        scale_ch= torch.sigmoid(avgout + maxout).unsqueeze(-1)  # (b,c,1)
        x= x* scale_ch
        # spatial attn
        avg_sp= torch.mean(x, dim=1, keepdim=True) # (b,1,t)
        max_sp,_= torch.max(x, dim=1, keepdim=True)
        cat_sp= torch.cat([avg_sp, max_sp], dim=1) # (b,2,t)
        scale_sp= torch.sigmoid(self.conv_spatial(cat_sp)) # (b,1,t)
        x= x* scale_sp
        return x

# ======================================
# (C) ResBlock
# ======================================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_cbam=False):
        super().__init__()
        self.use_cbam= use_cbam
        self.conv1= nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1= nn.BatchNorm1d(out_ch)
        self.conv2= nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2= nn.BatchNorm1d(out_ch)
        self.relu= nn.ReLU(inplace=True)
        self.downsample=None
        if stride!=1 or in_ch!= out_ch:
            self.downsample= nn.Sequential(
                nn.Conv1d(in_ch,out_ch,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm1d(out_ch)
            )
        if self.use_cbam:
            self.cbam= CBAM1D(out_ch)
    def forward(self, x):
        identity= x
        out= self.conv1(x)
        out= self.bn1(out)
        out= self.relu(out)
        out= self.conv2(out)
        out= self.bn2(out)
        if self.downsample is not None:
            identity= self.downsample(identity)
        out= out+ identity
        out= self.relu(out)
        if self.use_cbam:
            out= self.cbam(out)
        return out

# ======================================
# (D) DeepAttnBPModel
#  - wave(2,1250)-> conv -> ResBlock*N -> global avg pooling => wave_feat
#  - extra(6)-> MLP(多層) => extra_feat
#  - concat => hidden => 分頭 -> (sbp, dbp)
# ======================================
class DeepAttnBPModel(nn.Module):
    def __init__(self, base_ch=16, use_cbam=True):
        super().__init__()
        # conv1: in_ch=2 => out_ch= base_ch => stride=2
        self.conv1= nn.Conv1d(2, base_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1= nn.BatchNorm1d(base_ch)
        
        # resblocks
        self.res1= ResBlock1D(base_ch, base_ch, stride=1, use_cbam=use_cbam)
        self.res2= ResBlock1D(base_ch, base_ch*2, stride=2, use_cbam=use_cbam)
        self.res3= ResBlock1D(base_ch*2, base_ch*2, stride=1, use_cbam=use_cbam)
        
        # global avg pool => wave_feat( base_ch*2 )
        # (T會縮小 => 1250/2=625 -> /2= ~312 => stride=1 => 312 => GAvg => shape= (b, base_ch*2)
        self.gap= nn.AdaptiveAvgPool1d(1)

        # wave fc
        self.wave_fc= nn.Linear(base_ch*2, 64)  # wave_feat => 64

        # extra mlp (多層)
        self.extra_mlp= nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # fusion => 64 + 16= 80 => hidden => 32
        self.fusion_fc= nn.Linear(80,32)
        # 分頭 => sbp, dbp
        self.fc_sbp= nn.Linear(32,1)
        self.fc_dbp= nn.Linear(32,1)

    def forward(self, wave, extra):
        """
        wave: (B,2,1250), extra:(B,6)
        """
        x= self.conv1(wave)  # =>(B, base_ch, ~625)
        x= self.bn1(x)
        x= F.relu(x)

        x= self.res1(x)      # =>(B, base_ch, ~625)
        x= self.res2(x)      # =>(B, base_ch*2, ~312)
        x= self.res3(x)      # =>(B, base_ch*2, ~312)

        # global avg pool => (b, base_ch*2,1) => (b, base_ch*2)
        x= self.gap(x).squeeze(-1)  # =>(b, base_ch*2)
        x= self.wave_fc(x)         # =>(b,64)
        x= F.relu(x)

        e= self.extra_mlp(extra)   # =>(b,16)

        cat= torch.cat([x,e], dim=1)  # =>(b,80)
        cat= self.fusion_fc(cat)      # =>(b,32)
        cat= F.relu(cat)

        sbp= self.fc_sbp(cat)   # =>(b,1)
        dbp= self.fc_dbp(cat)   # =>(b,1)
        out= torch.cat([sbp, dbp], dim=1)  # =>(b,2)
        return out


# ======================================
# (E) 單一subject的訓練/驗證
# ======================================
def train_subject_model(dataset, model, num_epochs=10, batch_size=32, device='cpu'):
    """
    7:3 時序拆分 => train => val
    分別計算 SBP/DBP 的 loss / mae
    """
    model.to(device)
    N= len(dataset)
    if N<2:
        return None

    n_train= int(N*0.7)
    idxs= np.arange(N)
    i_train= idxs[:n_train]
    i_val= idxs[n_train:]
    ds_train= Subset(dataset, i_train)
    ds_val= Subset(dataset, i_val)

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)

    # 分開算 SBP, DBP
    # 這裡示範對 (sbp,dbp) 分別 L1Loss 加總 => overall loss
    def compute_loss(pred, label):
        # pred:(b,2), label:(b,2)
        sbp_pred= pred[:,0]
        dbp_pred= pred[:,1]
        sbp_label= label[:,0]
        dbp_label= label[:,1]
        loss_sbp= F.l1_loss(sbp_pred, sbp_label)
        loss_dbp= F.l1_loss(dbp_pred, dbp_label)
        return loss_sbp+ loss_dbp, loss_sbp.item(), loss_dbp.item()

    best_val= 999.0
    best_state= None

    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss=0; train_count=0
        train_sbp_mae=0; train_dbp_mae=0
        for wave, extra, label in train_loader:
            wave= wave.to(device)
            extra= extra.to(device)
            label= label.to(device)

            optimizer.zero_grad()
            pred= model(wave, extra)
            loss, sbp_l, dbp_l= compute_loss(pred,label)
            loss.backward()
            optimizer.step()

            bs= wave.size(0)
            train_loss+= loss.item()* bs
            train_sbp_mae+= sbp_l* bs
            train_dbp_mae+= dbp_l* bs
            train_count+= bs

        train_loss= train_loss/train_count
        train_sbp= train_sbp_mae/train_count
        train_dbp= train_dbp_mae/train_count

        # val
        model.eval()
        val_loss=0; val_count=0
        val_sbp_mae=0; val_dbp_mae=0
        with torch.no_grad():
            for wave, extra, label in val_loader:
                wave= wave.to(device)
                extra= extra.to(device)
                label= label.to(device)
                pred= model(wave, extra)
                loss, sbp_l, dbp_l= compute_loss(pred,label)
                bs= wave.size(0)
                val_loss+= loss.item()*bs
                val_sbp_mae+= sbp_l*bs
                val_dbp_mae+= dbp_l*bs
                val_count+= bs
        val_loss= val_loss/val_count
        val_sbp= val_sbp_mae/val_count
        val_dbp= val_dbp_mae/val_count
        if epoch%5==0:
            print(f"Epoch[{epoch+1}/{num_epochs}] "
              f"TrainLoss={train_loss:.3f}, SBP={train_sbp:.3f}, DBP={train_dbp:.3f}  |  "
              f"ValLoss={val_loss:.3f}, SBP={val_sbp:.3f}, DBP={val_dbp:.3f}")

        # 簡單 early best
        if val_loss< best_val:
            best_val= val_loss
            best_state= model.state_dict()

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final measure
    model.eval()
    # final train
    t_loss=0; t_count=0
    t_sbp=0; t_dbp=0
    with torch.no_grad():
        for wave, extra, label in train_loader:
            wave= wave.to(device)
            extra= extra.to(device)
            label= label.to(device)
            pred= model(wave, extra)
            loss, sbp_l, dbp_l= compute_loss(pred,label)
            bs= wave.size(0)
            t_loss+= loss.item()*bs
            t_sbp += sbp_l*bs
            t_dbp += dbp_l*bs
            t_count+= bs
    final_train_loss= t_loss/t_count
    final_train_sbp= t_sbp/t_count
    final_train_dbp= t_dbp/t_count

    # final val
    v_loss=0; v_count=0
    v_sbp=0; v_dbp=0
    with torch.no_grad():
        for wave, extra, label in val_loader:
            wave= wave.to(device)
            extra= extra.to(device)
            label= label.to(device)
            pred= model(wave, extra)
            loss, sbp_l, dbp_l= compute_loss(pred,label)
            bs= wave.size(0)
            v_loss+= loss.item()*bs
            v_sbp += sbp_l*bs
            v_dbp += dbp_l*bs
            v_count+= bs
    final_val_loss= v_loss/v_count
    final_val_sbp= v_sbp/v_count
    final_val_dbp= v_dbp/v_count

    return {
      'train_loss': final_train_loss,
      'train_sbp': final_train_sbp,
      'train_dbp': final_train_dbp,
      'val_loss': final_val_loss,
      'val_sbp': final_val_sbp,
      'val_dbp': final_val_dbp
    }

# ======================================
# (F) main
# ======================================
def main():
    device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device= {device}")

    # 指定資料夾
    h5_dir= "personalized_training_data_VitalDB"
    p= Path(h5_dir)
    all_h5= sorted(p.glob("*.h5"))

    model= DeepAttnBPModel(base_ch=16, use_cbam=True)
    params= sum(p.numel() for p in model.parameters())
    logger.info(f"Model params= {params}")
    results=[]
    for f in all_h5:
        subject_name= f.stem
        logger.info(f"\n=== 處理 {subject_name} ===")
        ds= SingleSubjectDataset(str(f))
        if len(ds)<2:
            logger.warning(f"{subject_name} => segments太少, skip.")
            continue
        
        # 初始化深度+attention+demographic模型
        logger.info(f"[{subject_name}] => segments= {len(ds)}")

        # 訓練
        stat= train_subject_model(ds, model, num_epochs=100, batch_size=32, device=device)
        if stat is None:
            logger.warning(f"{subject_name} => 無法訓練.")
            continue
        
        logger.info(f"[{subject_name}] => "
                    f"TrainLoss= {stat['train_loss']:.3f}, SBP= {stat['train_sbp']:.3f}, DBP= {stat['train_dbp']:.3f} | "
                    f"ValLoss= {stat['val_loss']:.3f}, SBP= {stat['val_sbp']:.3f}, DBP= {stat['val_dbp']:.3f}")
        # 收集
        row= {'subject': subject_name}
        row.update(stat)
        results.append(row)

    # 輸出 CSV
    import pandas as pd
    df_res= pd.DataFrame(results)
    out_csv= "personal_training_results_attention.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"已保存結果於 {out_csv}")

if __name__=="__main__":
    main()
