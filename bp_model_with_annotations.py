# ##################################################
# # conv_lstm_train.py
# ##################################################
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# import numpy as np
# import h5py
# from pathlib import Path
# from tqdm import tqdm
# import os

# ############################
# # (A) Dataset
# ############################
# class VitalSignDataset(Dataset):
#     """
#     從單一 .h5 檔案讀取資料:
#       ppg: shape=(N,1250)
#       ecg: shape=(N,1250)
#       annotations: shape=(N,1250,4)
#       personal_info: shape=(N,4)
#       vascular_properties: shape=(N,2)
#       segsbp, segdbp => label=(N,2)
#     """
#     def __init__(self, h5_file):
#         super().__init__()
#         self.h5 = h5py.File(h5_file, 'r')
#         self.ppg = self.h5['ppg']               # (N,1250)
#         self.ecg = self.h5['ecg']               # (N,1250)
#         self.annotations = self.h5['annotations']  # (N,1250,4)
#         self.personal_info = self.h5['personal_info']  # (N,4)
#         self.vascular = self.h5['vascular_properties'] # (N,2)
#         self.labels = np.stack([self.h5['segsbp'], self.h5['segdbp']], axis=1)  # (N,2)

#     def __len__(self):
#         return len(self.ppg)

#     def __getitem__(self, idx):
#         ppg_data  = self.ppg[idx]            # shape=(1250,)
#         ecg_data  = self.ecg[idx]            # shape=(1250,)
#         anno_data = self.annotations[idx]    # shape=(1250,4)
#         pers_info = self.personal_info[idx]  # shape=(4,)
#         vasc_info = self.vascular[idx]       # shape=(2,)
#         label_data= self.labels[idx]         # shape=(2,)

#         # torch化
#         ppg_t  = torch.from_numpy(ppg_data).float().unsqueeze(0)    # (1,1250)
#         ecg_t  = torch.from_numpy(ecg_data).float().unsqueeze(0)    # (1,1250)
#         anno_t = torch.from_numpy(anno_data).float().permute(1,0)   # (4,1250)
#         pers_t = torch.from_numpy(pers_info).float()   # (4,)
#         vasc_t = torch.from_numpy(vasc_info).float()   # (2,)
#         label_t= torch.from_numpy(label_data).float()  # (2,)

#         return ppg_t, ecg_t, anno_t, pers_t, vasc_t, label_t


# ############################
# # (B) ConvLSTMBPEstimator
# ############################
# class ConvLSTMBPEstimator(nn.Module):
#     """
#     1) CNN: (ppg+ecg+anno=6 ch) => conv => pool => shape=(B, C, reduced_len)
#     2) permute => (B, reduced_len, C) => LSTM => output=(B,hidden_dim)
#     3) concat with personal(4) & vascular(2) embed => final => (SBP, DBP)
#     """
#     def __init__(self, hidden_dim=32):
#         super().__init__()
#         self.input_channels = 6  # ppg(1) + ecg(1) + anno(4)
        
#         # CNN: conv1->conv2->pool
#         self.conv1 = nn.Conv1d(self.input_channels, 32, kernel_size=5, padding=2)
#         self.bn1   = nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#         self.bn2   = nn.BatchNorm1d(64)
#         self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)  # 1250->625

#         # LSTM
#         self.lstm  = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)

#         # personal(4)->16, vascular(2)->16
#         self.fc_personal = nn.Linear(4, 16)
#         self.fc_vascular = nn.Linear(2, 16)

#         # final => hidden_dim(32) + personal(16) + vascular(16) => 64 => (2)
#         self.fc_final= nn.Linear(hidden_dim+16+16, 2)

#     def forward(self, ppg, ecg, anno, pers_info, vasc_info):
#         # wave=(B,6,1250)
#         wave = torch.cat([ppg, ecg, anno], dim=1)

#         # CNN
#         x = F.relu(self.bn1(self.conv1(wave)))  # =>(B,32,1250)
#         x = F.relu(self.bn2(self.conv2(x)))     # =>(B,64,1250)
#         x = self.pool(x)                        # =>(B,64,625)

#         # LSTM expects (B, seq_len, feature_dim)
#         x = x.permute(0,2,1)                    # =>(B,625,64)
#         out_lstm, (hn, cn) = self.lstm(x)
#         # 取最後 time step => (B, hidden_dim)
#         feat_seq = out_lstm[:,-1,:]

#         # personal, vascular embed
#         pers_e = F.relu(self.fc_personal(pers_info))  # =>(B,16)
#         vasc_e = F.relu(self.fc_vascular(vasc_info))  # =>(B,16)

#         comb = torch.cat([feat_seq, pers_e, vasc_e], dim=1)  # =>(B, hidden_dim+16+16)
#         out  = self.fc_final(comb)                          # =>(B,2)
#         return out


# ############################
# # (C) Evaluate & Train
# ############################
# def evaluate_model_mae(model, dataloader, device='cpu'):
#     model.eval()
#     total_mae=0.0
#     total_count=0
#     with torch.no_grad():
#         for ppg, ecg, anno, pers, vasc, labels in dataloader:
#             ppg= ppg.to(device)
#             ecg= ecg.to(device)
#             anno= anno.to(device)
#             pers= pers.to(device)
#             vasc= vasc.to(device)
#             labels= labels.to(device)

#             preds= model(ppg, ecg, anno, pers, vasc)
#             mae_sbp= torch.sum(torch.abs(preds[:,0]- labels[:,0]))
#             mae_dbp= torch.sum(torch.abs(preds[:,1]- labels[:,1]))
#             batch_size= ppg.size(0)
#             total_mae += (mae_sbp+ mae_dbp).item()
#             total_count+= batch_size*2
#     return total_mae/ total_count

# def train_model(model, dataloaders, optimizer, num_epochs=50, device='cpu'):
#     criterion= nn.L1Loss()
#     init_val_mae= evaluate_model_mae(model, dataloaders['val'], device=device)
#     print(f"[Initial] val MAE= {init_val_mae:.4f}")
#     init_test_mae= evaluate_model_mae(model, dataloaders['test'], device=device)
#     print(f"[Initial] test MAE= {init_test_mae:.4f}")

#     best_wts= model.state_dict()
#     best_mae= init_val_mae

#     for epoch in range(1, num_epochs+1):
#         print(f"Epoch {epoch}/{num_epochs}")
#         for phase in ['train','val']:
#             if phase=='train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss=0.0
#             total_samples=0

#             for ppg, ecg, anno, pers, vasc, labels in tqdm(dataloaders[phase]):
#                 ppg= ppg.to(device)
#                 ecg= ecg.to(device)
#                 anno= anno.to(device)
#                 pers= pers.to(device)
#                 vasc= vasc.to(device)
#                 labels= labels.to(device)

#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(phase=='train'):
#                     preds= model(ppg, ecg, anno, pers, vasc)
#                     loss= criterion(preds, labels)
#                     if phase=='train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item()* ppg.size(0)
#                 total_samples += ppg.size(0)

#             epoch_loss= running_loss/ total_samples
#             print(f"{phase} MAE= {epoch_loss:.4f}")

#             if phase=='val' and epoch_loss< best_mae:
#                 best_mae= epoch_loss
#                 best_wts= model.state_dict()
#         print("-"*30)

#     print(f"Training done, best val MAE= {best_mae:.4f}")
#     test_mae= evaluate_model_mae(model, dataloaders['test'], device=device)
#     print(f"Test MAE= {test_mae:.4f}")
#     model.load_state_dict(best_wts)
#     return model


# ############################
# # (D) main
# ############################
# if __name__=='__main__':
#     data_dir= Path('training_data_VitalDB_quality')
#     train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
#     val_file= data_dir/'validation.h5'
#     test_file= data_dir/'test.h5'

#     from torch.utils.data import ConcatDataset
#     train_dss=[]
#     for tf in train_files:
#         if tf.exists():
#             train_dss.append(VitalSignDataset(str(tf)))
#     train_dataset= ConcatDataset(train_dss)
#     val_dataset= VitalSignDataset(str(val_file))
#     test_dataset= VitalSignDataset(str(test_file))

#     train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
#     val_loader  = DataLoader(val_dataset,   batch_size=32, shuffle=False, drop_last=False)
#     test_loader = DataLoader(test_dataset,  batch_size=32, shuffle=False, drop_last=False)
#     dataloaders= {'train': train_loader, 'val': val_loader, 'test': test_loader}

#     device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # 初始化 Conv-LSTM 模型
#     model= ConvLSTMBPEstimator(hidden_dim=32).to(device)
#     if os.path.exists("bp_estimator_conv_lstm6.pth"):
#         model.load_state_dict(torch.load("bp_estimator_conv_lstm6.pth"))
#         print("Model loaded from bp_estimator_conv_lstm6.pth")
#     #參數量
#     total_params= sum(p.numel() for p in model.parameters())
#     input(f"Total parameters: {total_params}")

#     optimizer= optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
#     model= train_model(model, dataloaders, optimizer, num_epochs=100, device=device)

#     torch.save(model.state_dict(), "bp_estimator_conv_lstm6.pth")
#     print("[Done] Model saved.")

##################################################
# conv_lstm_train.py (改成 "深層 + Attention" 版本)
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

############################
# (A) Dataset (維持不變)
############################
class VitalSignDataset(Dataset):
    def __init__(self, h5_file):
        super().__init__()
        self.h5 = h5py.File(h5_file, 'r')
        self.ppg = self.h5['ppg']               # (N,1250)
        self.ecg = self.h5['ecg']               # (N,1250)
        self.annotations = self.h5['annotations']  # (N,1250,4)
        self.personal_info = self.h5['personal_info']  # (N,4)
        self.vascular = self.h5['vascular_properties'] # (N,2)
        self.labels = np.stack([self.h5['segsbp'], self.h5['segdbp']], axis=1)  # (N,2)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        ppg_data  = self.ppg[idx]            # shape=(1250,)
        ecg_data  = self.ecg[idx]            # shape=(1250,)
        anno_data = self.annotations[idx]    # shape=(1250,4)
        pers_info = self.personal_info[idx]  # shape=(4,)
        vasc_info = self.vascular[idx]       # shape=(2,)
        label_data= self.labels[idx]         # shape=(2,)

        # torch化
        ppg_t  = torch.from_numpy(ppg_data).float().unsqueeze(0)    # (1,1250)
        ecg_t  = torch.from_numpy(ecg_data).float().unsqueeze(0)    # (1,1250)
        anno_t = torch.from_numpy(anno_data).float().permute(1,0)   # (4,1250)
        pers_t = torch.from_numpy(pers_info).float()   # (4,)
        vasc_t = torch.from_numpy(vasc_info).float()   # (2,)
        label_t= torch.from_numpy(label_data).float()  # (2,)

        return ppg_t, ecg_t, anno_t, pers_t, vasc_t, label_t


############################
# (B) DeepConvAttnBPEstimator (新模型)
############################
class DeepConvAttnBPEstimator(nn.Module):
    """
    更深層 + Attention 機制:
      - 2 個 Conv Blocks，每個 block: Conv->BN->ReLU->Conv->BN->ReLU->MaxPool
      - LSTM 處理 time dim
      - Attention 機制取得加權後 context
      - 與 Personal / Vascular 各自的 Embed 結合後，經 MLP => 輸出 (SBP, DBP)
    """
    def __init__(self, hidden_dim=64, attn_dim=64):
        super().__init__()
        self.input_channels = 6  # ppg(1) + ecg(1) + anno(4)
        
        # --- Conv Block1 ---
        self.conv1_1 = nn.Conv1d(self.input_channels, 32, kernel_size=5, padding=2)
        self.bn1_1   = nn.BatchNorm1d(32)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.bn1_2   = nn.BatchNorm1d(32)
        self.pool1   = nn.MaxPool1d(kernel_size=2, stride=2)  # 1250 -> 625

        # --- Conv Block2 ---
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2_1   = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2_2   = nn.BatchNorm1d(64)
        self.pool2   = nn.MaxPool1d(kernel_size=2, stride=2)  # 625 -> 312

        # LSTM
        self.lstm  = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        # 如果想用雙向 LSTM，可以 bidirectional=True，記得後面維度會 x2

        # Attention
        self.attn_linear = nn.Linear(hidden_dim, 1)  # 簡單的 "score = W*h_t"
        # 若 bidirectional=True, hidden_dim要改成 2*hidden_dim

        # personal(4)->16, vascular(2)->16
        self.fc_personal = nn.Linear(4, 16)
        self.fc_vascular = nn.Linear(2, 16)

        # MLP Output
        # 最後輸入維度= hidden_dim + 16 + 16
        self.fc_hidden = nn.Linear(hidden_dim + 16 + 16, 64)
        self.fc_out    = nn.Linear(64, 2)

    def forward(self, ppg, ecg, anno, pers_info, vasc_info):
        # wave = (B,6,1250)
        wave = torch.cat([ppg, ecg, anno], dim=1)

        # --- block1 ---
        x = F.relu(self.bn1_1(self.conv1_1(wave)))   # =>(B,32,1250)
        x = F.relu(self.bn1_2(self.conv1_2(x)))      # =>(B,32,1250)
        x = self.pool1(x)                            # =>(B,32,625)

        # --- block2 ---
        x = F.relu(self.bn2_1(self.conv2_1(x)))      # =>(B,64,625)
        x = F.relu(self.bn2_2(self.conv2_2(x)))      # =>(B,64,625)
        x = self.pool2(x)                            # =>(B,64,312)

        # LSTM expects (B, seq_len, feature_dim)
        x = x.permute(0, 2, 1)                       # =>(B,312,64)
        out_lstm, (hn, cn) = self.lstm(x)            # out_lstm: (B,312,hidden_dim)

        # Attention: 對 out_lstm 的每個 time step 做打分數 => (B,312,1)，再 softmax => 加權求和
        # e = attn_linear(out_lstm) => (B,312,1)
        e = self.attn_linear(out_lstm)               # =>(B,312,1)
        alpha = F.softmax(e, dim=1)                  # =>(B,312,1)
        # 加權和 => context
        context = torch.sum(out_lstm * alpha, dim=1) # =>(B, hidden_dim)

        # personal, vascular embed
        pers_e = F.relu(self.fc_personal(pers_info)) # =>(B,16)
        vasc_e = F.relu(self.fc_vascular(vasc_info)) # =>(B,16)

        # combine
        comb = torch.cat([context, pers_e, vasc_e], dim=1)  
        # =>(B, hidden_dim + 16 + 16)

        # MLP => (2)
        hidden = F.relu(self.fc_hidden(comb))        # =>(B,64)
        out    = self.fc_out(hidden)                 # =>(B,2)
        return out


############################
# (C) Evaluate & Train (幾乎不變)
############################
def evaluate_model_mae(model, dataloader, device='cpu'):
    model.eval()
    total_mae=0.0
    total_count=0
    with torch.no_grad():
        for ppg, ecg, anno, pers, vasc, labels in dataloader:
            ppg= ppg.to(device)
            ecg= ecg.to(device)
            anno= anno.to(device)
            pers= pers.to(device)
            vasc= vasc.to(device)
            labels= labels.to(device)

            preds= model(ppg, ecg, anno, pers, vasc)
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
    init_test_mae= evaluate_model_mae(model, dataloaders['test'], device=device)
    print(f"[Initial] test MAE= {init_test_mae:.4f}")

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

            for ppg, ecg, anno, pers, vasc, labels in tqdm(dataloaders[phase]):
                ppg= ppg.to(device)
                ecg= ecg.to(device)
                anno= anno.to(device)
                pers= pers.to(device)
                vasc= vasc.to(device)
                labels= labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    preds= model(ppg, ecg, anno, pers, vasc)
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
    test_mae= evaluate_model_mae(model, dataloaders['test'], device=device)
    print(f"Test MAE= {test_mae:.4f}")
    model.load_state_dict(best_wts)
    return model


############################
# (D) main
############################
if __name__=='__main__':
    data_dir= Path('training_data_VitalDB_quality')
    train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
    val_file= data_dir/'validation.h5'
    test_file= data_dir/'test.h5'

    from torch.utils.data import ConcatDataset
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

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 使用深層 + Attention 的新模型
    model= DeepConvAttnBPEstimator(hidden_dim=64, attn_dim=64).to(device)
    # 如果要載入舊權重，可以再視需求調整 key/結構

    # 檢查參數量
    total_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    optimizer= optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    model= train_model(model, dataloaders, optimizer, num_epochs=100, device=device)

    torch.save(model.state_dict(), "bp_estimator_deep_conv_attn.pth")
    print("[Done] Model saved.")
