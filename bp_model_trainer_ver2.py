import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from torchviz import make_dot  # 用於視覺化
from prepare_training_data import calculate_first_derivative, calculate_second_derivative, encode_personal_info
from torch.optim.lr_scheduler import ReduceLROnPlateau

######################################
# 1) FiLM Block (中途融合)
######################################
class FiLMBlock(nn.Module):
    def __init__(self, embed_dim, num_channels):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(embed_dim, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )
        self.beta_fc  = nn.Sequential(
            nn.Linear(embed_dim, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x, info_emb):
        """
        x: (B, C, L)       - wave feature map
        info_emb: (B, E)   - personal_info embedding (e.g. 16 or 32 dim)
        """
        gamma = self.gamma_fc(info_emb)  # (B, C)
        beta  = self.beta_fc(info_emb)   # (B, C)
        gamma = gamma.unsqueeze(-1)      # => (B, C, 1)
        beta  = beta.unsqueeze(-1)       # => (B, C, 1)

        return gamma * x + beta

######################################
# 2) Multi-Head Attention Pooling
######################################
class MultiHeadAttnPooling(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, d_model))  # learnable query
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, C, L), 需要 C == d_model
        回傳: (B, C)
        """
        B, C, L = x.shape
        if C != self.d_model:
            raise ValueError(f"Channel {C} != d_model {self.d_model}, need alignment.")

        # (B,L,C)
        x_t = x.transpose(1,2)
        # (B,1,C)
        query = self.query.unsqueeze(0).expand(B, -1, -1)

        out, _ = self.mha(query, x_t, x_t)  # => (B,1,C)
        out = out.squeeze(1)               # => (B,C)
        out = self.ln(out)
        return out

######################################
# 3) Dataset: 支援 1250 點長度
######################################
class MultiH5Dataset(Dataset):
    def __init__(self, list_of_h5):
        self.datasets = []
        for h5_file in list_of_h5:
            ds = BPDataset(h5_file)
            self.datasets.append(ds)
        self._lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self._lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        offset = 0
        for ds_idx, ds in enumerate(self.datasets):
            size = self._lengths[ds_idx]
            if idx < offset + size:
                return ds[idx - offset]
            offset += size
        raise IndexError("Index out of range.")

class BPDataset(Dataset):
    """
    讀取 1250 點長度訊號 (ppg, ecg, vpg, apg)，外加 SBP/DBP 與 personal_info
    """
    def __init__(self, h5_path):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            # print(f"\n[Dataset] 載入資料: {self.h5_path}")
            # for k in f.keys():
            #     print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

            self.ppg = torch.FloatTensor(f['ppg'][:])     # (N,1250)
            self.ecg = torch.FloatTensor(f['ecg'][:])     # (N,1250)
            self.segsbp = torch.FloatTensor(f['segsbp'][:])  # (N,)
            self.segdbp = torch.FloatTensor(f['segdbp'][:])  # (N,)
            self.bp_2d = torch.stack([self.segsbp, self.segdbp], dim=1)  # => (N,2)

            if 'personal_info' in f:
                self.personal_info = torch.FloatTensor(f['personal_info'][:]) # (N,M)
            else:
                N = self.ppg.shape[0]
                self.personal_info = torch.zeros((N, 4))

            # 如果有 vpg, apg 就直接讀；沒有就由 ppg 算
            if 'vpg' in f:
                self.vpg = torch.FloatTensor(f['vpg'][:])
            else:
                self.vpg = self.calculate_first_derivative(self.ppg)

            if 'apg' in f:
                self.apg = torch.FloatTensor(f['apg'][:])
            else:
                self.apg = self.calculate_second_derivative(self.ppg)

        # (N,1250) => (N,1,1250)
        self.ppg = self.ppg.unsqueeze(1)
        self.vpg = self.vpg.unsqueeze(1)
        self.apg = self.apg.unsqueeze(1)
        self.ecg = self.ecg.unsqueeze(1)

    def calculate_first_derivative(self, signal):
        # shape=(N,1250)
        vpg = torch.zeros_like(signal)
        vpg[:,1:-1] = (signal[:,2:] - signal[:,:-2]) / 2
        vpg[:,0] = vpg[:,1]
        vpg[:,-1] = vpg[:,-2]
        return vpg

    def calculate_second_derivative(self, signal):
        # shape=(N,1250)
        apg = torch.zeros_like(signal)
        apg[:,1:-1] = signal[:,2:] - 2*signal[:,1:-1] + signal[:,:-2]
        apg[:,0] = apg[:,1]
        apg[:,-1] = apg[:,-2]
        return apg

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx],
            'vpg': self.vpg[idx],
            'apg': self.apg[idx],
            'ecg': self.ecg[idx],
            'bp_values': self.bp_2d[idx],
            'personal_info': self.personal_info[idx]
        }

######################################
# 4) ResBlock1D
######################################
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu= nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.downsample = (stride!=1 or in_ch!=out_ch)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        sc = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += sc
        out = self.relu(out)
        return out

######################################
# 5) DeepResBranchNetFiLM
######################################
class DeepResBranchNetFiLM(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, layers_per_stage=[2,2,2,2],
                 film_embed_dim=16):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = self._make_stage(base_filters,     base_filters,     layers_per_stage[0], stride=1)
        self.stage2 = self._make_stage(base_filters,     base_filters*2,   layers_per_stage[1], stride=2)
        self.stage3 = self._make_stage(base_filters*2,   base_filters*4,   layers_per_stage[2], stride=2)
        self.stage4 = self._make_stage(base_filters*4,   base_filters*8,   layers_per_stage[3], stride=2)

        # FiLM
        self.film1 = FiLMBlock(film_embed_dim, base_filters)
        self.film2 = FiLMBlock(film_embed_dim, base_filters*2)
        self.film3 = FiLMBlock(film_embed_dim, base_filters*4)
        self.film4 = FiLMBlock(film_embed_dim, base_filters*8)

        self.out_ch = base_filters*8

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks=[]
        blocks.append(ResBlock1D(in_ch, out_ch, stride=stride))
        for _ in range(num_blocks-1):
            blocks.append(ResBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x, info_emb):
        x = self.conv_in(x)      # => (B, base_filters, 1250/4=~312 依 stride)
        x = self.stage1(x)
        x = self.film1(x, info_emb)

        x = self.stage2(x)
        x = self.film2(x, info_emb)

        x = self.stage3(x)
        x = self.film3(x, info_emb)

        x = self.stage4(x)
        x = self.film4(x, info_emb)
        return x

######################################
# 6) 最終模型 BPEstimator
######################################
class BPEstimator(nn.Module):
    def __init__(self, info_dim=5, base_filters=32, layers_per_stage=[2,2,2,2],
                 film_embed_dim=16, d_model_attn=256, n_heads=4):
        super().__init__()

        # personal info => FiLM
        self.info_fc_film = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU(),
            nn.Linear(32, film_embed_dim),
            nn.ReLU()
        )

        self.ppg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )
        self.vpg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )
        self.apg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )
        self.ecg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )

        self.branch_out_ch = base_filters*8  # e.g. 256
        if self.branch_out_ch != d_model_attn:
            self.align_conv = nn.Conv1d(self.branch_out_ch, d_model_attn, 1, bias=False)
        else:
            self.align_conv = None

        self.attn_pool_ppg = MultiHeadAttnPooling(d_model_attn, n_heads)
        self.attn_pool_vpg = MultiHeadAttnPooling(d_model_attn, n_heads)
        self.attn_pool_apg = MultiHeadAttnPooling(d_model_attn, n_heads)
        self.attn_pool_ecg = MultiHeadAttnPooling(d_model_attn, n_heads)

        self.info_fc_final = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU()
        )

        final_in_dim = d_model_attn*4 + 32
        self.final_fc = nn.Sequential(
            nn.Linear(final_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, ppg, vpg, apg, ecg, personal_info):
        # FiLM embedding
        film_emb = self.info_fc_film(personal_info)  # (B, film_embed_dim)
        # 4 branches
        ppg_map = self.ppg_branch(ppg, film_emb)
        vpg_map = self.vpg_branch(vpg, film_emb)
        apg_map = self.apg_branch(apg, film_emb)
        ecg_map = self.ecg_branch(ecg, film_emb)

        # align if needed
        if self.align_conv is not None:
            ppg_map = self.align_conv(ppg_map)
            vpg_map = self.align_conv(vpg_map)
            apg_map = self.align_conv(apg_map)
            ecg_map = self.align_conv(ecg_map)

        # MHA Pool
        ppg_feat = self.attn_pool_ppg(ppg_map)
        vpg_feat = self.attn_pool_vpg(vpg_map)
        apg_feat = self.attn_pool_apg(apg_map)
        ecg_feat = self.attn_pool_ecg(ecg_map)

        # final info
        info_emb_final = self.info_fc_final(personal_info)

        # concat
        wave_cat = torch.cat([ppg_feat, vpg_feat, apg_feat, ecg_feat], dim=1)
        combined = torch.cat([wave_cat, info_emb_final], dim=1)
        out = self.final_fc(combined)
        return out

class BPEstimatorNoECG(nn.Module):
    """
    不使用 ECG 分支 => 只保留 PPG, VPG, APG 三路 DeepResBranchNetFiLM + personal_info
    """
    def __init__(self, info_dim=5, base_filters=32, layers_per_stage=[2,2,2,2],
                 film_embed_dim=16, d_model_attn=256, n_heads=4):
        super().__init__()
        # personal info => FiLM
        self.info_fc_film = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU(),
            nn.Linear(32, film_embed_dim),
            nn.ReLU()
        )

        # 三個分支: ppg, vpg, apg
        self.ppg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )
        self.vpg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )
        self.apg_branch = DeepResBranchNetFiLM(
            in_channels=1, base_filters=base_filters,
            layers_per_stage=layers_per_stage,
            film_embed_dim=film_embed_dim
        )

        self.branch_out_ch = base_filters*8  # e.g. 256
        if self.branch_out_ch != d_model_attn:
            self.align_conv = nn.Conv1d(self.branch_out_ch, d_model_attn, 1, bias=False)
        else:
            self.align_conv = None

        # 3 個分支 => MHA pool
        self.attn_pool_ppg = MultiHeadAttnPooling(d_model_attn, n_heads)
        self.attn_pool_vpg = MultiHeadAttnPooling(d_model_attn, n_heads)
        self.attn_pool_apg = MultiHeadAttnPooling(d_model_attn, n_heads)

        # personal info (再次embedding for final concat)
        self.info_fc_final = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU()
        )

        # 最終 concat => (3*d_model_attn + 32)
        final_in_dim = d_model_attn*3 + 32
        self.final_fc = nn.Sequential(
            nn.Linear(final_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, ppg, vpg, apg, personal_info):
        """
        沒有 ecg branch
        ppg, vpg, apg: shape=(B,1,1250)
        personal_info: shape=(B, info_dim)
        """
        # 1) FiLM embedding
        film_emb = self.info_fc_film(personal_info)  # (B, film_embed_dim)

        # 2) 三路 wave
        ppg_map = self.ppg_branch(ppg, film_emb)
        vpg_map = self.vpg_branch(vpg, film_emb)
        apg_map = self.apg_branch(apg, film_emb)

        # 3) align if needed
        if self.align_conv is not None:
            ppg_map = self.align_conv(ppg_map)
            vpg_map = self.align_conv(vpg_map)
            apg_map = self.align_conv(apg_map)

        # 4) MHA pool => (B, d_model_attn)
        ppg_feat = self.attn_pool_ppg(ppg_map)
        vpg_feat = self.attn_pool_vpg(vpg_map)
        apg_feat = self.attn_pool_apg(apg_map)

        # 5) final info => 32
        info_emb_final = self.info_fc_final(personal_info)  # (B,32)

        # 6) concat => (B, 3*d_model_attn + 32)
        wave_cat = torch.cat([ppg_feat, vpg_feat, apg_feat], dim=1)
        combined = torch.cat([wave_cat, info_emb_final], dim=1)

        # 7) final fc => (B,2) => [SBP,DBP]
        out = self.final_fc(combined)
        return out


######################################
# 7) 訓練流程
######################################
class BPTrainer:
    def __init__(self, fold_path, device='cuda', batch_size=32, lr=1e-3):
        self.fold_path = Path(fold_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr

    def create_dataloaders(self):
        train_files = [self.fold_path/f"training_{i}.h5" for i in range(1,10)]
        train_set = MultiH5Dataset(train_files)

        val_set = BPDataset(self.fold_path/'validation.h5')
        test_set= BPDataset(self.fold_path/'test.h5')

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 檢查一個 batch
        sample_batch = next(iter(train_loader))
        print("\n[Check one train batch]")
        for k,v in sample_batch.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        return train_loader, val_loader, test_loader

    def visualize_model(self, model, output_path="model_structure_1250.png"):
        # 建立 dummy input: (1,1,1250)
        dummy_ppg = torch.randn(1, 1, 1250).to(self.device)
        dummy_vpg = torch.randn(1, 1, 1250).to(self.device)
        dummy_apg = torch.randn(1, 1, 1250).to(self.device)
        dummy_ecg = torch.randn(1, 1, 1250).to(self.device)
        dummy_info= torch.randn(1,5).to(self.device)

        out = model(dummy_ppg, dummy_vpg, dummy_apg, dummy_ecg, dummy_info)
        dot = make_dot(out, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render(output_path, cleanup=True)
        print(f"[INFO] 模型結構圖保存於 {output_path}")

    def train(self, epochs=50, early_stop_patience=10):

        model = BPEstimator(
            info_dim=5,
            base_filters=32,
            layers_per_stage=[2,2,2,2],
            film_embed_dim=16,
            d_model_attn=256,
            n_heads=4
        ).to(self.device)

        # 視覺化
        # self.visualize_model(model, "model_structure_1250")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Model] total={total_params:,}, trainable={trainable_params:,}\n")

        train_loader, val_loader, test_loader = self.create_dataloaders()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs+1):
            # training
            model.train()
            running_loss = 0.0
            running_mae  = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values = batch['bp_values'].to(self.device)
                pers_info = batch['personal_info'].to(self.device)

                optimizer.zero_grad()
                preds = model(ppg, vpg, apg, ecg, pers_info)
                loss = criterion(preds, bp_values)
                mae  = torch.mean(torch.abs(preds - bp_values))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_mae  += mae.item()

            train_loss = running_loss/len(train_loader)
            train_mae  = running_mae/len(train_loader)

            # validation
            model.eval()
            val_loss=0.0
            val_mae=0.0
            val_me=0.0
            with torch.no_grad():
                for batch in val_loader:
                    ppg = batch['ppg'].to(self.device)
                    vpg = batch['vpg'].to(self.device)
                    apg = batch['apg'].to(self.device)
                    ecg = batch['ecg'].to(self.device)
                    bp_values=batch['bp_values'].to(self.device)
                    pers_info=batch['personal_info'].to(self.device)

                    preds = model(ppg, vpg, apg, ecg, pers_info)
                    l = criterion(preds,bp_values)
                    m = torch.mean(torch.abs(preds - bp_values))
                    e = torch.mean(preds - bp_values)
                    val_loss += l.item()
                    val_mae  += m.item()
                    val_me   += e.item()
            val_loss/=len(val_loader)
            val_mae /=len(val_loader)
            val_me  /=len(val_loader)
            scheduler.step(val_loss)

            if val_loss<best_val_loss:
                best_val_loss= val_loss
                patience_counter=0
                torch.save(model.state_dict(), self.fold_path/"best_model_1250.pt")
            else:
                patience_counter+=1
                if patience_counter>= early_stop_patience:
                    print(f"[Info] Early stop at epoch={epoch}")
                    break

            print(f"[Epoch {epoch}/{epochs}] "
                  f"TrainLoss={train_loss:.4f},MAE={train_mae:.4f} | "
                  f"ValLoss={val_loss:.4f},MAE={val_mae:.4f},ME={val_me:.4f}")

        # testing
        print("\n=== Testing ===")
        model.load_state_dict(torch.load(self.fold_path/"best_model_1250.pt"))
        model.eval()
        test_loss=0.0
        test_mae=0.0
        with torch.no_grad():
            for batch in test_loader:
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values=batch['bp_values'].to(self.device)
                pers_info=batch['personal_info'].to(self.device)

                preds=model(ppg, vpg, apg, ecg, pers_info)
                l=criterion(preds,bp_values)
                m=torch.mean(torch.abs(preds - bp_values))

                test_loss+= l.item()
                test_mae += m.item()

        test_loss/=len(test_loader)
        test_mae /=len(test_loader)
        print(f"Test MSE={test_loss:.4f}, MAE={test_mae:.4f}")
        return model


class BPTrainerNoECG:
    def __init__(self, fold_path, device='cuda', batch_size=32, lr=1e-3):
        self.fold_path = Path(fold_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr

    def create_dataloaders(self):
        train_files = [self.fold_path/f"training_{i}.h5" for i in range(1,10)]
        from torch.utils.data import DataLoader
        train_set = MultiH5Dataset(train_files)

        val_set   = BPDataset(self.fold_path/'validation.h5')
        test_set  = BPDataset(self.fold_path/'test.h5')

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 檢查一個 batch
        sample_batch = next(iter(train_loader))
        print("\n[Check one train batch - NoECG model]")
        for k,v in sample_batch.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        return train_loader, val_loader, test_loader

    def train(self, epochs=50, early_stop_patience=10):
        """
        與原本 train 流程類似，但改成 BPEstimatorNoECG
        並在 forward 時不再用 ecg
        """
        model = BPEstimatorNoECG(
            info_dim=5,
            base_filters=32,
            layers_per_stage=[2,2,2,2],
            film_embed_dim=16,
            d_model_attn=256,
            n_heads=4
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        input(f"\n[Model] total={total_params:,}, trainable={trainable_params:,}\n")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        train_loader, val_loader, test_loader = self.create_dataloaders()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs+1):
            # ---- training ----
            model.train()
            running_loss = 0.0
            running_mae  = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                ppg = batch['ppg'].to(self.device)   # (B,1,1250)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                # ecg = batch['ecg'].to(self.device)  # 不用
                pers_info = batch['personal_info'].to(self.device)
                bp_values = batch['bp_values'].to(self.device) # (B,2)

                optimizer.zero_grad()
                preds = model(ppg, vpg, apg, pers_info)  # 無 ecg
                loss = criterion(preds, bp_values)
                mae  = torch.mean(torch.abs(preds - bp_values))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_mae  += mae.item()

            train_loss = running_loss/len(train_loader)
            train_mae  = running_mae/len(train_loader)

            # ---- validation ----
            model.eval()
            val_loss = 0.0
            val_mae  = 0.0
            val_me   = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    ppg = batch['ppg'].to(self.device)
                    vpg = batch['vpg'].to(self.device)
                    apg = batch['apg'].to(self.device)
                    pers_info = batch['personal_info'].to(self.device)
                    bp_values = batch['bp_values'].to(self.device)

                    preds = model(ppg, vpg, apg, pers_info)
                    l = criterion(preds, bp_values)
                    m = torch.mean(torch.abs(preds - bp_values))
                    e = torch.mean(preds - bp_values) # mean error

                    val_loss += l.item()
                    val_mae  += m.item()
                    val_me   += e.item()

            val_loss /= len(val_loader)
            val_mae  /= len(val_loader)
            val_me   /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.fold_path/"best_model_noecg.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"[Info] Early stopping at epoch={epoch}")
                    break

            print(f"[Epoch {epoch}/{epochs}] TrainLoss={train_loss:.4f}, MAE={train_mae:.4f} | "
                  f"ValLoss={val_loss:.4f}, MAE={val_mae:.4f}, ME={val_me:.4f}")

        # ---- testing ----
        print("\n=== Testing (NoECG) ===")
        model.load_state_dict(torch.load(self.fold_path/"best_model_noecg.pt"))
        model.eval()
        test_loss=0.0
        test_mae =0.0
        with torch.no_grad():
            for batch in test_loader:
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                pers_info = batch['personal_info'].to(self.device)
                bp_values= batch['bp_values'].to(self.device)

                preds = model(ppg, vpg, apg, pers_info)
                l=criterion(preds,bp_values)
                m=torch.mean(torch.abs(preds - bp_values))

                test_loss+= l.item()
                test_mae += m.item()

        test_loss/=len(test_loader)
        test_mae /=len(test_loader)
        print(f"[NoECG] Test MSE={test_loss:.4f}, MAE={test_mae:.4f}")

        return model

if __name__=='__main__':
    trainer = BPTrainer(
        fold_path= 'training_data_1250_MIMIC_test',  #'training_data_1250',
        device='cuda',
        batch_size=64,  # 視記憶體可改
        lr=1e-3
    )
    final_model = trainer.train(epochs=50, early_stop_patience=10)
# if __name__=='__main__':
#     trainer_noecg = BPTrainerNoECG(
#         fold_path='training_data_1250',
#         device='cuda',
#         batch_size=32,And how? Yeah. Hmm. Ah. Hmm. Batman. That. No.
#         lr=1e-3
#     )
#     final_model_noecg = trainer_noecg.train(epochs=50, early_stop_patience=10)


def convert_single_rawfile_to_processed(in_path, out_path):
    """
    從 原始 .mat.h5 (含 PPG_Raw, ECG_Raw, ABP_Raw, SegSBP, SegDBP, Age,BMI,Gender,Height,Weight...)
    轉成 與 BPDataset 相容的 .h5
    
    - 每個 segment => (1250,) ppg,ecg => (1250,) => vpg,apg
    - segsbp, segdbp => scalar
    - personal_info => 5 維 [Age,BMI,Gender,Height,Weight], 以 encode_personal_info 編碼
    - 寫到 out_path => 欄位: 'ppg','ecg','vpg','apg','segsbp','segdbp','personal_info'
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    with h5py.File(in_path, 'r') as f_in:
        # ---- 讀個人資訊 (可能 shape=(1,), 需要 flatten)
        keys = ['Age','BMI','Gender','Height','Weight']
        person_dict = {}
        for k in keys:
            if k in f_in:
                arr = f_in[k][:]
                val = float(arr.flatten()[0]) if arr.size>0 else 0.0
                person_dict[k] = val
            else:
                person_dict[k] = 0.0
        personal_info_enc = encode_personal_info(person_dict)

        # ---- 讀 raw wave
        if 'PPG_Raw' not in f_in or 'ECG_Raw' not in f_in or 'ABP_Raw' not in f_in:
            print(f"[Warning] {in_path} missing PPG_Raw/ECG_Raw/ABP_Raw, skip.")
            return False
        
        ppg_raw = f_in['PPG_Raw'][:]  # shape=(n_segments,1250)
        ecg_raw = f_in['ECG_Raw'][:]
        abp_raw = f_in['ABP_Raw'][:]
        
        segsbp = f_in['SegSBP'][:]     # shape=(n_segments,1)
        segdbp = f_in['SegDBP'][:]

        n_segments = len(ppg_raw)
        data_list = []
        
        # ---- 逐 segment 處理
        for i in range(n_segments):
            ppg_1d = ppg_raw[i] # (1250,)
            ecg_1d = ecg_raw[i]
            abp_1d = abp_raw[i]

            # segsbp[i], segdbp[i] 可能 shape=(1,)
            sbp_val = float(segsbp[i][0])  # or segsbp[i]
            dbp_val = float(segdbp[i][0])
            
            # 計算 vpg, apg
            vpg_1d = calculate_first_derivative(ppg_1d)
            apg_1d = calculate_second_derivative(ppg_1d)
            
            # ---- 檢查合不合理 (ex: sbp>=20? dbp>=20?)
            # 或者您可以再檢查 abp_1d max/min
            if sbp_val>300 or sbp_val<20 or dbp_val>300 or dbp_val<20:
                # skip
                continue
            
            # ---- 存成 dict
            data_list.append({
                'ppg': ppg_1d,
                'vpg': vpg_1d,
                'apg': apg_1d,
                'ecg': ecg_1d,
                'segsbp': sbp_val,
                'segdbp': dbp_val,
                'personal_info': personal_info_enc
            })
    
    if len(data_list)==0:
        print(f"[Warning] {in_path} => no valid segments, skip.")
        return False
    
    # ---- 寫出 out_path
    n_samples = len(data_list)
    info_dim = len(data_list[0]['personal_info'])
    input_len = 1250

    with h5py.File(out_path, 'w') as f_out:
        f_out.create_dataset('ppg',     (n_samples, input_len), dtype='float32')
        f_out.create_dataset('vpg',     (n_samples, input_len), dtype='float32')
        f_out.create_dataset('apg',     (n_samples, input_len), dtype='float32')
        f_out.create_dataset('ecg',     (n_samples, input_len), dtype='float32')
        f_out.create_dataset('segsbp',  (n_samples,), dtype='float32')
        f_out.create_dataset('segdbp',  (n_samples,), dtype='float32')
        f_out.create_dataset('personal_info', (n_samples, info_dim), dtype='float32')

        for i, dic in enumerate(data_list):
            f_out['ppg'][i]    = dic['ppg']
            f_out['vpg'][i]    = dic['vpg']
            f_out['apg'][i]    = dic['apg']
            f_out['ecg'][i]    = dic['ecg']
            f_out['segsbp'][i] = dic['segsbp']
            f_out['segdbp'][i] = dic['segdbp']
            f_out['personal_info'][i] = dic['personal_info']
    
    print(f"[Info] convert_single_rawfile_to_processed => {out_path}, N={n_samples}")
    return True

class BPDatasetSingle(torch.utils.data.Dataset):
    """ 與您原本 BPDataset 幾乎相同，只是我們這裡獨立寫一個簡化版 """
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path,'r') as f:
            self.ppg = torch.from_numpy(f['ppg'][:])         # shape=(N,1250)
            self.vpg = torch.from_numpy(f['vpg'][:])
            self.apg = torch.from_numpy(f['apg'][:])
            self.ecg = torch.from_numpy(f['ecg'][:])
            self.bp_2d = torch.from_numpy(
                np.stack([f['segsbp'][:], f['segdbp'][:]], axis=1)
            )   # (N,2)
            self.personal_info = torch.from_numpy(f['personal_info'][:])  # (N,5) or (N,4)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx].unsqueeze(0),  # shape=(1,1250)
            'vpg': self.vpg[idx].unsqueeze(0),
            'apg': self.apg[idx].unsqueeze(0),
            'ecg': self.ecg[idx].unsqueeze(0),
            'bp_values': self.bp_2d[idx],       # (2,)
            'personal_info': self.personal_info[idx]
        }

def evaluate_single_h5(model, h5_path, device='cuda', batch_size=32):
    """
    讀取處理後的 h5 (含ppg,vpg,apg,ecg,segsbp,segdbp,personal_info)
    用 batch_size 跑，回傳整檔案的MAE
    """
    ds = BPDatasetSingle(h5_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dl:
            ppg = batch['ppg'].to(device)
            vpg = batch['vpg'].to(device)
            apg = batch['apg'].to(device)
            ecg = batch['ecg'].to(device)
            bp_values = batch['bp_values'].to(device)
            pers_info = batch['personal_info'].to(device)

            preds = model(ppg, vpg, apg, ecg, pers_info)  # (B,2)
            abs_err = torch.abs(preds - bp_values)
            sum_abs_err = abs_err.sum().item()
            
            total_mae += sum_abs_err
            total_count += abs_err.numel()  # B*2

    mean_mae = total_mae / total_count
    return mean_mae

def evaluate_single_rawfile(model, rawfile_path, tmp_processed_path, device='cuda'):
    """
    先將 rawfile_path (原始 .mat.h5) => convert_single_rawfile_to_processed => tmp_processed_path
    再用 evaluate_single_h5 計算 MAE
    """
    success = convert_single_rawfile_to_processed(rawfile_path, tmp_processed_path)
    if not success:
        print(f"[Error] convert single {rawfile_path} failed.")
        return None
    
    mae = evaluate_single_h5(model, tmp_processed_path, device=device, batch_size=32)
    return mae

def evaluate_test_files_individually(model, files_txt, data_dir, device='cuda'):
    """
    讀 test_files.txt (每行一個原始檔 .mat.h5)
    對每個檔做 evaluate_single_rawfile(...), 最後打印平均MAE
    """
    data_dir = Path(data_dir)
    with open(files_txt, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    results = []
    for fname in file_list:
        raw_path = data_dir / fname  # e.g. processed_data/xxx.mat.h5
        if not raw_path.exists():
            print(f"[Warn] {raw_path} not found, skip.")
            continue
        
        # 要有個暫存 out_path
        tmp_out = raw_path.with_name(fname.replace('.mat.h5','_processed.h5'))
        
        mae = evaluate_single_rawfile(model, raw_path, tmp_out, device=device)
        if mae is not None:
            results.append((fname, mae))
            print(f"[Single File] {fname}, MAE={mae:.4f}")
        else:
            print(f"[Fail] {fname}")
    
    if results:
        overall_mae = sum(mae for (_, mae) in results) / len(results)
        print(f"\n[Overall] across {len(results)} files, mean MAE={overall_mae:.4f}")
    else:
        print("[Info] No valid files found or all failed.")



# if __name__=='__main__':
#     # 1) 先建模 & load state_dict
#     model = BPEstimator(
#         info_dim=5,
#         base_filters=32,
#         layers_per_stage=[2,2,2,2],
#         film_embed_dim=16,
#         d_model_attn=256,
#         n_heads=4
#     )
#     device = torch.device('cuda')
#     model.to(device)
    
#     # load 已訓練好的權重 (e.g. best_model_1250.pt)
#     checkpoint_path = 'training_data_1250/best_model_1250.pt'
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()
    
#     # 2) 逐檔名測試
#     # 假設 test_files.txt 路徑: e.g. "training_data_1250/test_files.txt"
#     #    processed_data 資料夾: e.g. "processed_data/"
#     test_files_txt = 'training_data_1250/test_files.txt'
#     data_dir = 'processed_data'  # 裝原始處理後 .h5
#     evaluate_test_files_individually(model, test_files_txt, data_dir, device=device)