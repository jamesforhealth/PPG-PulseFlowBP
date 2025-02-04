"""
MAML_ResNet_BP.py

目標:
 - 使用 1D ResNet + CBAM 簡單版, 融合 demographic features
 - MAML 流程 (使用 higher), 從多受試者task學到可快速適應的初始權重
 - 希望 base model val loss ~10, few-shot calibration => ~5
安裝:
 pip install higher

資料前提:
 - 每檔 .h5 (e.g. subjectX.h5) 含:
    ppg: (N,1250)
    ecg: (N,1250)
    personal_info: (N,4)
    vascular_properties: (N,2)
    segsbp, segdbp: (N,)
 - path => h5_dir/subject001.h5, ...
 - 一檔 => 一人 => SingleSubjectDataset => (support, query)
 
程式流程:
1) build_tasks_list(h5_dir)
2) meta_train_maml(... ResNet model ...)
3) 測試: new subject => few-shot => check MAE
"""

import os
import numpy as np
import random
from pathlib import Path
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import higher
from tqdm import tqdm

# ---------------------------
# (A) Dataset
# ---------------------------
class SingleSubjectDataset(Dataset):
    """
    讀取單一受試者 .h5 => (wave, extra, label)
    wave: (C,1250)
    extra: (6,) (personal(4)+vascular(2))
    label: (2, ) => (sbp, dbp)
    """
    def __init__(self, h5_path):
        super().__init__()
        self.f = h5py.File(h5_path,'r')
        self.ppg = self.f['ppg'][:]   # (N,1250)
        self.ecg = self.f['ecg'][:]
        self.sbp = self.f['segsbp'][:]
        self.dbp = self.f['segdbp'][:]
        self.pers= self.f['personal_info'][:]
        self.vasc= self.f['vascular_properties'][:]
        self.N= len(self.ppg)

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # wave => e.g. 2 ch (ppg, ecg)
        ppg_ = self.ppg[idx]   # (1250,)
        ecg_ = self.ecg[idx]   # (1250,)
        wave_2ch= np.stack([ppg_, ecg_], axis=0)  # shape=(2,1250)

        # extra => shape=(6,)
        pers_ = self.pers[idx]   # (4,)
        vasc_ = self.vasc[idx]   # (2,)
        extra_ = np.concatenate([pers_, vasc_], axis=0)  # =>(6,)

        # label => (2,) => sbp,dbp
        sbp_= self.sbp[idx]
        dbp_= self.dbp[idx]
        label_= np.array([sbp_, dbp_], dtype=np.float32)

        # to tensor
        wave_t= torch.from_numpy(wave_2ch).float()
        extra_t= torch.from_numpy(extra_).float()
        label_t= torch.from_numpy(label_).float()
        return wave_t, extra_t, label_t


def load_subject_as_task(h5_path, support_ratio=0.5):
    ds = SingleSubjectDataset(h5_path)
    n = len(ds)
    if n <2: return None
    n_support= int(n* support_ratio)
    idxs= list(range(n))
    random.shuffle(idxs)
    i_supp= idxs[:n_support]
    i_query= idxs[n_support:]
    if len(i_supp)<1 or len(i_query)<1:
        return None
    ds_supp= Subset(ds, i_supp)
    ds_query= Subset(ds, i_query)
    return (ds_supp, ds_query)

def build_tasks_list(h5_dir, support_ratio=0.5):
    p= Path(h5_dir)
    files= list(p.glob("*.h5"))
    tasks=[]
    for f in tqdm(files, desc="build tasks"):
        t= load_subject_as_task(str(f), support_ratio)
        if t is not None:
            tasks.append(t)
    return tasks

# ---------------------------
# (B) 1D ResNet + CBAM + demographic fusion
# ---------------------------

class CBAM1D(nn.Module):
    """簡易 Channel+Spatial注意力 (1D)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channels= channels
        self.reduction= reduction

        # Channel attn
        self.avg_pool= nn.AdaptiveAvgPool1d(1)
        self.max_pool= nn.AdaptiveMaxPool1d(1)
        self.fc= nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False)
        )
        # Spatial attn
        self.conv_spatial= nn.Conv1d(2,1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # x: (B,C,T)
        b,c,t= x.shape
        # channel
        avgout= self.fc(self.avg_pool(x).view(b,c))
        maxout= self.fc(self.max_pool(x).view(b,c))
        scale_ch= torch.sigmoid(avgout + maxout).unsqueeze(-1)  # (b,c,1)
        x= x * scale_ch

        # spatial
        avg_spatial= torch.mean(x, dim=1, keepdim=True) # (b,1,t)
        max_spatial,_= torch.max(x, dim=1, keepdim=True)
        cat_spatial= torch.cat([avg_spatial, max_spatial], dim=1) # (b,2,t)
        scale_sp= torch.sigmoid(self.conv_spatial(cat_spatial))   # (b,1,t)
        x= x* scale_sp
        return x

class ResBlock1D(nn.Module):
    """1D ResBlock with optional stride + CBAM"""
    def __init__(self, in_ch, out_ch, stride=1, use_cbam=False):
        super().__init__()
        self.use_cbam= use_cbam
        self.conv1= nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1= nn.BatchNorm1d(out_ch)
        self.conv2= nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2= nn.BatchNorm1d(out_ch)
        self.relu= nn.ReLU(inplace=True)
        self.downsample=None
        if stride!=1 or in_ch!=out_ch:
            self.downsample= nn.Sequential(
                nn.Conv1d(in_ch,out_ch, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm1d(out_ch)
            )
        if self.use_cbam:
            self.cbam= CBAM1D(out_ch)

    def forward(self,x):
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

class DeepResBPModel(nn.Module):
    """
    ex:
    wave => shape=(B,2,1250). 
    - encode by 2~3 layers ResBlock => flatten => fc
    - demographic(6) => linear => concat
    - final => regress 2 => (SBP,DBP)
    """
    def __init__(self, base_ch=32):
        super().__init__()
        # 1) conv1 => in_ch=2 => out_ch=base_ch => stride=1
        self.conv1= nn.Conv1d(2, base_ch, kernel_size=3, padding=1,bias=False)
        self.bn1= nn.BatchNorm1d(base_ch)
        
        # 2) stack resblocks
        self.res1= ResBlock1D(base_ch, base_ch, stride=1, use_cbam=False)
        self.res2= ResBlock1D(base_ch, base_ch*2, stride=2, use_cbam=True)  # downsample => T/2
        self.res3= ResBlock1D(base_ch*2, base_ch*2, stride=1, use_cbam=False)
        # 以上可再加更多

        # demographic => fc
        self.dem_fc= nn.Linear(6, base_ch)  # =>(b,base_ch)

        # flatten => fc hidden => out=2
        self.fc_hidden= nn.Linear(base_ch*2*(1250//2), 128)  
        # 這裡(1250//2)表示最後一個res2 stride=2 => T=625 => out_ch= base_ch*2 => c*(t)= (64*625=40000)...
        # 可能很大 => 需檢視 GPU

        self.fc_dem= nn.Linear(base_ch + 128, 64)  
        self.fc_out= nn.Linear(64,2)

    def forward(self, wave, extra):
        """
        wave: (B,2,1250)
        extra:(B,6)
        """
        b,c,t= wave.shape
        x= self.conv1(wave)
        x= self.bn1(x)
        x= F.relu(x)

        x= self.res1(x)  # =>(b, base_ch, t)
        x= self.res2(x)  # =>(b, base_ch*2, t/2)
        x= self.res3(x)  # =>(b, base_ch*2, t/2)

        # flatten
        x= x.view(b,-1)   # =>(b, base_ch*2*(t/2))
        # fc hidden
        x= self.fc_hidden(x)
        x= F.relu(x)

        # demographic
        dem= self.dem_fc(extra)
        dem= F.relu(dem)

        # concat
        cat= torch.cat([x, dem], dim=1)
        cat= F.relu(self.fc_dem(cat))

        out= self.fc_out(cat)  # =>(b,2)
        return out


# ---------------------------
# (C) MAML
# ---------------------------
def inner_adapt(model, optimizer, support_loader, inner_steps, device):
    model.train()
    with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True) as (fmodel, diffopt):
        for _ in range(inner_steps):
            for wave, extra, label in support_loader:
                wave= wave.to(device)
                extra= extra.to(device)
                label= label.to(device)
                pred= fmodel(wave, extra)
                loss= F.l1_loss(pred, label)
                diffopt.step(loss)
        return fmodel

def meta_train_maml(tasks, model, outer_opt, device='cpu',
                    meta_batch_size=4, 
                    inner_lr=1e-3,
                    inner_steps=1,
                    outer_steps=50):
    """
    tasks: [(ds_supp, ds_query), ...]
    model: deep resnet
    outer_opt: Adam
    meta_batch_size=4 => each outer step sample 4 tasks
    inner_lr => used for inner sgd
    """
    model.to(device)
    for step in range(outer_steps):
        # sample tasks
        batch_tasks= random.sample(tasks, meta_batch_size)
        outer_opt.zero_grad()

        meta_loss=0.0
        for ds_supp, ds_query in batch_tasks:
            supp_loader= DataLoader(ds_supp, batch_size=8, shuffle=True)
            query_loader= DataLoader(ds_query, batch_size=8, shuffle=False)
            
            # inner opt
            inner_opt= torch.optim.SGD(model.parameters(), lr=inner_lr)
            
            # inner loop
            fmodel= inner_adapt(model, inner_opt, supp_loader, inner_steps, device)

            # compute query loss => accumulate
            q_loss= 0.0
            q_count= 0
            for wave, extra, label in query_loader:
                wave= wave.to(device)
                extra= extra.to(device)
                label= label.to(device)
                pred= fmodel(wave, extra)
                loss= F.l1_loss(pred, label)
                bs= wave.size(0)
                q_loss+= (loss*bs)
                q_count+= bs
            q_loss= q_loss/q_count
            q_loss.backward()
            meta_loss+= q_loss.item()
        meta_loss= meta_loss/ meta_batch_size
        outer_opt.step()

        print(f"[outer step={step}] meta_loss= {meta_loss:.4f}")


# ---------------------------
# (D) main
# ---------------------------
def main():
    device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("using device=", device)

    # 1) 準備 h5_dir
    h5_dir= "personalized_training_data_VitalDB"
    tasks= build_tasks_list(h5_dir, support_ratio=0.5)
    print(f"共 {len(tasks)} 個Task(受試者)")

    # 2) init deep model
    model= DeepResBPModel(base_ch=32)

    # 3) outer opt
    outer_opt= torch.optim.Adam(model.parameters(), lr=1e-4)

    # 4) meta train
    meta_train_maml(tasks, model, outer_opt,
                    device=device,
                    meta_batch_size=4,
                    inner_lr=1e-3,
                    inner_steps=1,
                    outer_steps=100)

    # 5) save
    torch.save(model.state_dict(), "maml_deep_resnet_cbam.pth")
    print("MAML meta-train完成, 權重已保存.")


if __name__=="__main__":
    main()
