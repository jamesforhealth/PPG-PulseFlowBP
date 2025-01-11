import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import h5py
from tqdm import tqdm
import pandas as pd
import numpy as np

from bp_model_trainer_ver2 import BPEstimator, BPDataset
# ↑ 假設你的 BPEstimator, BPDataset 在這個檔案中。
# 如果你原本不是這樣import，請自行調整。

class PersonalizedSegmentTrainer:
    def __init__(self, 
                 pretrained_model_path, 
                 personal_data_dir, 
                 device='cuda', 
                 batch_size=1,
                 segment_counts=None):
        """
        segment_counts: list[int], e.g. [1,5,10,20,30,50] 
                        要嘗試 fine-tune 時取用的訓練segment數量
        """
        self.pretrained_path = Path(pretrained_model_path)
        self.personal_data_dir = Path(personal_data_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size
        if segment_counts is None:
            # 預設測試1,5,10,20,50 ...
            segment_counts = [1, 2, 3, 5, 10, 20, 50]
        self.segment_counts = segment_counts
        
        # 檢查檔案路徑
        if not self.pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {self.pretrained_path}")
        if not self.personal_data_dir.exists():
            raise FileNotFoundError(f"Personal data directory not found at {self.personal_data_dir}")
        
        self.results_df = pd.DataFrame(columns=[
            'subject_id', 
            'n_segments', 
            'original_val_mae', 
            'finetuned_val_mae',
            'improvement'
        ])

    def load_pretrained_model(self):
        """載入預訓練模型 (BPEstimator) 並凍結多數層，只開啟最後若干層。"""
        model = BPEstimator(
            info_dim=5,
            base_filters=32,
            layers_per_stage=[2,2,2,2],
            film_embed_dim=16,
            d_model_attn=256,
            n_heads=4
        ).to(self.device)
        
        state_dict = torch.load(self.pretrained_path)
        # 不做特殊過濾就整個載入
        model.load_state_dict(state_dict)
        
        # 這裡如果你只想開啟 final_fc/attn_pool 來fine-tune，可改以下
        for param in model.parameters():
            param.requires_grad = False
        
        # 解凍(可學習)的參數
        def unfreeze_params(module):
            for p in module.parameters():
                p.requires_grad = True

        unfreeze_params(model.final_fc)
        unfreeze_params(model.attn_pool_ppg)
        unfreeze_params(model.attn_pool_vpg)
        unfreeze_params(model.attn_pool_apg)
        unfreeze_params(model.attn_pool_ecg)
        unfreeze_params(model.info_fc_final)

        # 打印可訓練參數數量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n[load_pretrained_model] trainable params: {trainable_params:,}/{total_params:,}")
        return model

    def create_optimizer(self, model):
        # 如果想對不同層賦予不同lr，可使用 params_dict
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4, weight_decay=1e-4
        )
        return optimizer
    
    def evaluate_model(self, model, dataloader):
        model.eval()
        total_mae_sbp = 0.0
        total_mae_dbp = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values = batch['bp_values'].float().to(self.device)
                pers_info = batch['personal_info'].float().to(self.device)

                preds = model(ppg, vpg, apg, ecg, pers_info)
                mae_sbp = torch.sum(torch.abs(preds[:, 0] - bp_values[:, 0]))
                mae_dbp = torch.sum(torch.abs(preds[:, 1] - bp_values[:, 1]))
                
                batch_size = ppg.size(0)
                total_mae_sbp += mae_sbp.item()
                total_mae_dbp += mae_dbp.item()
                total_samples += batch_size
        
        avg_mae_sbp = total_mae_sbp / total_samples
        avg_mae_dbp = total_mae_dbp / total_samples
        avg_mae = (avg_mae_sbp + avg_mae_dbp) / 2.0
        return avg_mae

    def finetune_for_subject(self, subject_id, n_segments, max_epochs=30, early_stop_patience=7):
        """
        只取該 subject 前 n_segments 筆訓練資料 來 fine-tune
        """
        train_path = self.personal_data_dir/f"{subject_id}_train.h5"
        val_path   = self.personal_data_dir/f"{subject_id}_val.h5"
        if (not train_path.exists()) or (not val_path.exists()):
            print(f"[Skip] train/val not found for subject {subject_id}")
            return None, None
        
        # 載入資料
        train_ds = BPDataset(train_path)
        val_ds   = BPDataset(val_path)
        
        # 如果 n_segments 大於該subject的 train_ds 總長，就只取實際有的
        actual_n = min(n_segments, len(train_ds))
        # Subset 只取前 actual_n 筆
        subset_indices = list(range(actual_n))
        train_subset = Subset(train_ds, subset_indices)
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,       batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 載入預訓練模型
        model = self.load_pretrained_model()
        # 先在Val set上做evaluation => original
        original_val_mae = self.evaluate_model(model, val_loader)
        print(f"[Initial] subject={subject_id}, n_segments={n_segments}, original_val_mae={original_val_mae:.4f}")

        # 建立 optimizer
        criterion = nn.MSELoss()
        optimizer = self.create_optimizer(model)
        # 你可考慮CosineAnnealingLR, 這裡示範用ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, min_lr=1e-6)

        # fine-tune
        best_val_mae = float('inf')
        best_state_dict = None
        patience_cnt = 0

        for epoch in range(1, max_epochs+1):
            model.train()
            running_mae = 0.0
            running_count = 0
            for batch in train_loader:
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values = batch['bp_values'].float().to(self.device)
                pers_info = batch['personal_info'].float().to(self.device)

                optimizer.zero_grad()
                preds = model(ppg, vpg, apg, ecg, pers_info)
                
                loss_sbp = criterion(preds[:,0], bp_values[:,0])
                loss_dbp = criterion(preds[:,1], bp_values[:,1])
                loss = (loss_sbp + loss_dbp)/2.0
                
                mae_sbp = torch.mean(torch.abs(preds[:,0]-bp_values[:,0]))
                mae_dbp = torch.mean(torch.abs(preds[:,1]-bp_values[:,1]))
                mae = (mae_sbp + mae_dbp)/2.0

                loss.backward()
                optimizer.step()
                
                running_mae += mae.item()
                running_count += 1

            train_mae = running_mae / running_count if running_count>0 else 999.0
            # val
            val_mae = self.evaluate_model(model, val_loader)

            scheduler.step(val_mae)

            # early-stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state_dict = model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= early_stop_patience:
                    print(f"[EarlyStop] subject={subject_id}, n_segments={n_segments}, epoch={epoch}")
                    break
            
            print(f"[subject={subject_id}, seg={n_segments}] epoch={epoch}, train_mae={train_mae:.4f}, val_mae={val_mae:.4f}")

        # load best
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        finetuned_val_mae = self.evaluate_model(model, val_loader)

        return original_val_mae, finetuned_val_mae
    
    def get_all_subject_ids(self):
        """透過 personal_data_dir 下 _train.h5 檔案找出所有 subject_ids"""
        train_files = list(self.personal_data_dir.glob("*_train.h5"))
        subject_ids = [f.stem.replace("_train", "") for f in train_files]
        return sorted(subject_ids)

    def run_all_subjects(self):
        subject_ids = self.get_all_subject_ids()
        if not subject_ids:
            print(f"No subject found in {self.personal_data_dir}")
            return
        
        for sid in subject_ids:
            print(f"\n===== Subject {sid} =====")
            # 依序嘗試 self.segment_counts
            for nseg in self.segment_counts:
                print(f"\n--- Fine-tune with n_segments={nseg} ---")
                original_val_mae, finetuned_val_mae = self.finetune_for_subject(sid, nseg)
                if (original_val_mae is not None) and (finetuned_val_mae is not None):
                    improvement = original_val_mae - finetuned_val_mae
                    self.results_df.loc[len(self.results_df)] = {
                        'subject_id': sid,
                        'n_segments': nseg,
                        'original_val_mae': original_val_mae,
                        'finetuned_val_mae': finetuned_val_mae,
                        'improvement': improvement
                    }
        
        # 最後輸出
        if not self.results_df.empty:
            out_csv = self.personal_data_dir/"personal_finetune_by_segment.csv"
            self.results_df.to_csv(out_csv, index=False)
            print(f"\n[Done] see {out_csv}")

# 主程式
if __name__=="__main__":
    trainer = PersonalizedSegmentTrainer(
        pretrained_model_path="training_data_1250_MIMIC_test/best_model_1250.pt",
        personal_data_dir="personalized_training_data_MIMIC",
        device="cuda",
        batch_size=1,
        segment_counts= [1, 3, 5, 10, 20, 30, 40, 50]
    )
    trainer.run_all_subjects()
