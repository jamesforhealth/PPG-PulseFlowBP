##################################################
# conv_lstm_train.py
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

############################
# (A) Dataset
############################
class VitalSignDataset(Dataset):
    """
    從單一 .h5 檔案讀取資料:
      ppg: shape=(N,1250)
      annotations: shape=(N,1250,4)
      personal_info: shape=(N,4)
      vascular_properties: shape=(N,2)
      segsbp, segdbp => label=(N,2)
    """
    def __init__(self, h5_file):
        super().__init__()
        self.h5 = h5py.File(h5_file, 'r')
        self.ppg = self.h5['ppg']               # (N,1250)
        self.annotations = self.h5['annotations']  # (N,1250,4)
        self.personal_info = self.h5['personal_info']  # (N,4)
        self.vascular = self.h5['vascular_properties'] # (N,2)
        self.labels = np.stack([self.h5['segsbp'], self.h5['segdbp']], axis=1)  # (N,2)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        ppg_data  = self.ppg[idx]            # shape=(1250,)
        anno_data = self.annotations[idx]    # shape=(1250,4)
        pers_info = self.personal_info[idx]  # shape=(4,)
        vasc_info = self.vascular[idx]       # shape=(2,)
        label_data= self.labels[idx]         # shape=(2,)

        # torch化
        ppg_t  = torch.from_numpy(ppg_data).float().unsqueeze(0)    # (1,1250)
        anno_t = torch.from_numpy(anno_data).float().permute(1,0)   # (4,1250)
        pers_t = torch.from_numpy(pers_info).float()   # (4,)
        vasc_t = torch.from_numpy(vasc_info).float()   # (2,)
        label_t= torch.from_numpy(label_data).float()  # (2,)

        return ppg_t, anno_t, pers_t, vasc_t, label_t


############################
# (B) ConvLSTMBPEstimator
############################
class ConvLSTMBPEstimator(nn.Module):
    """
    1) CNN: (ppg+anno=5 ch) => conv => pool => shape=(B, C, reduced_len)
    2) permute => (B, reduced_len, C) => LSTM => output=(B,hidden_dim)
    3) concat with personal(4) & vascular(2) embed => final => (SBP, DBP)
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.input_channels = 5  # ppg(1) + anno(4)
        
        # CNN: conv1->conv2->pool
        self.conv1 = nn.Conv1d(self.input_channels, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)  # 1250->625

        # LSTM
        self.lstm  = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)

        # personal(4)->16, vascular(2)->16
        self.fc_personal = nn.Linear(4, 16)
        self.fc_vascular = nn.Linear(2, 16)

        # final => hidden_dim(32) + personal(16) + vascular(16) => 64 => (2)
        self.fc_final= nn.Linear(hidden_dim+16+16, 2)

    def forward(self, ppg, anno, pers_info, vasc_info):
        # wave=(B,5,1250)
        wave = torch.cat([ppg, anno], dim=1)

        # CNN
        x = F.relu(self.bn1(self.conv1(wave)))  # =>(B,32,1250)
        x = F.relu(self.bn2(self.conv2(x)))     # =>(B,64,1250)
        x = self.pool(x)                        # =>(B,64,625)

        # LSTM expects (B, seq_len, feature_dim)
        x = x.permute(0,2,1)                    # =>(B,625,64)
        out_lstm, (hn, cn) = self.lstm(x)
        # 取最後 time step => (B, hidden_dim)
        feat_seq = out_lstm[:,-1,:]

        # personal, vascular embed
        pers_e = F.relu(self.fc_personal(pers_info))  # =>(B,16)
        vasc_e = F.relu(self.fc_vascular(vasc_info))  # =>(B,16)

        comb = torch.cat([feat_seq, pers_e, vasc_e], dim=1)  # =>(B, hidden_dim+16+16)
        out  = self.fc_final(comb)                          # =>(B,2)
        return out


############################
# (C) Evaluate & Train
############################
def evaluate_model_mae(model, dataloader, device='cpu'):
    model.eval()
    total_mae=0.0
    total_count=0
    with torch.no_grad():
        for ppg, anno, pers, vasc, labels in dataloader:
            ppg= ppg.to(device)
            anno= anno.to(device)
            pers= pers.to(device)
            vasc= vasc.to(device)
            labels= labels.to(device)

            preds= model(ppg, anno, pers, vasc)
            mae_sbp= torch.sum(torch.abs(preds[:,0]- labels[:,0]))
            mae_dbp= torch.sum(torch.abs(preds[:,1]- labels[:,1]))
            batch_size= ppg.size(0)
            total_mae += (mae_sbp+ mae_dbp).item()
            total_count+= batch_size*2
    return total_mae/ total_count

def train_model(model, dataloaders, optimizer, num_epochs=50, device='cpu'):
    criterion= nn.L1Loss()
    init_val_mae= evaluate_model_mae(model, dataloaders['val'], device=device)
    print(f"[Initial] val MAE= {init_val_mae:.4f}")

    best_wts= model.state_dict()
    best_mae= init_val_mae

    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        for phase in ['train','val']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss=0.0
            total_samples=0

            for ppg, anno, pers, vasc, labels in tqdm(dataloaders[phase]):
                ppg= ppg.to(device)
                anno= anno.to(device)
                pers= pers.to(device)
                vasc= vasc.to(device)
                labels= labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    preds= model(ppg, anno, pers, vasc)
                    loss= criterion(preds, labels)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()* ppg.size(0)
                total_samples += ppg.size(0)

            epoch_loss= running_loss/ total_samples
            print(f"{phase} MAE= {epoch_loss:.4f}")

            if phase=='val' and epoch_loss< best_mae:
                best_mae= epoch_loss
                best_wts= model.state_dict()
        print("-"*30)

    print(f"Training done, best val MAE= {best_mae:.4f}")
    model.load_state_dict(best_wts)
    return model


############################
# (D) main
############################
if __name__=='__main__':
    data_dir= Path('training_data_VitalDB')
    train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
    val_file= data_dir/'validation.h5'

    from torch.utils.data import ConcatDataset
    train_dss=[]
    for tf in train_files:
        if tf.exists():
            train_dss.append(VitalSignDataset(str(tf)))
    train_dataset= ConcatDataset(train_dss)
    val_dataset= VitalSignDataset(str(val_file))

    train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader  = DataLoader(val_dataset,   batch_size=32, shuffle=False, drop_last=False)
    dataloaders= {'train': train_loader, 'val': val_loader}

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化 Conv-LSTM 模型
    model= ConvLSTMBPEstimator(hidden_dim=32).to(device)
    #參數量
    total_params= sum(p.numel() for p in model.parameters())
    input(f"Total parameters: {total_params}")

    optimizer= optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model= train_model(model, dataloaders, optimizer, num_epochs=50, device=device)

    torch.save(model.state_dict(), "bp_estimator_conv_lstm.pth")
    print("[Done] Model saved.")
