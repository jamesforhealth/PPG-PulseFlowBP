import os
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from bp_model_trainer_ver2 import BPDataset, BPEstimator

class PersonalizedEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """載入預訓練模型"""
        model = BPEstimator(
            info_dim=5,
            base_filters=32,
            layers_per_stage=[2,2,2,2],
            film_embed_dim=16,
            d_model_attn=256,
            n_heads=4
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def evaluate_person(self, h5_path):
        """評估單一個人的預測效果"""
        dataset = BPDataset(h5_path)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 取得輸入資料
                ppg = batch['ppg'].to(self.device)
                vpg = batch['vpg'].to(self.device)
                apg = batch['apg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                bp_values = batch['bp_values'].to(self.device)
                personal_info = batch['personal_info'].to(self.device)
                
                # 模型預測
                preds = self.model(ppg, vpg, apg, ecg, personal_info)
                
                # 收集預測結果
                all_preds.append(preds.cpu().numpy())
                all_targets.append(bp_values.cpu().numpy())
        
        # 合併所有批次的結果
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 計算評估指標
        mae = np.mean(np.abs(all_preds - all_targets), axis=0)
        rmse = np.sqrt(np.mean((all_preds - all_targets)**2, axis=0))
        me = np.mean(all_preds - all_targets, axis=0)
        std = np.std(all_preds - all_targets, axis=0)
        
        return {
            'n_samples': len(all_preds),
            'sbp_mae': mae[0],
            'dbp_mae': mae[1],
            'sbp_rmse': rmse[0],
            'dbp_rmse': rmse[1],
            'sbp_me': me[0],
            'dbp_me': me[1],
            'sbp_std': std[0],
            'dbp_std': std[1],
            'predictions': all_preds,
            'targets': all_targets
        }

def main():
    # 設定路徑
    test_list_path = "training_data_1250_MIMIC_test/test_files.txt"
    personalized_data_dir = Path("personalized_training_data_MIMIC")
    model_path = Path("training_data_1250_MIMIC_test/best_model_1250.pt")
    
    # 讀取測試檔案列表
    with open(test_list_path, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    # 建立評估器
    evaluator = PersonalizedEvaluator(model_path)
    
    # 儲存所有結果
    all_results = {}
    
    # 評估每個人
    print("\n開始評估每個人的預測效果...")
    for mat_file in tqdm(test_files):
        person_id = mat_file.replace('.mat', '')
        h5_path = personalized_data_dir / f"{person_id}_val.h5"
        
        if not h5_path.exists():
            print(f"警告: 找不到 {h5_path}")
            continue
        
        results = evaluator.evaluate_person(h5_path)
        all_results[person_id] = results
    
    # 輸出結果
    print("\n個人預測效果統計:")
    print("=" * 80)
    print(f"{'Person ID':12} {'N':>8} {'SBP MAE':>10} {'DBP MAE':>10} {'SBP RMSE':>10} {'DBP RMSE':>10}")
    print("-" * 80)
    
    total_samples = 0
    weighted_sbp_mae = 0
    weighted_dbp_mae = 0
    weighted_sbp_rmse = 0
    weighted_dbp_rmse = 0
    
    for person_id, results in all_results.items():
        n = results['n_samples']
        print(f"{person_id:12} {n:8d} {results['sbp_mae']:10.2f} {results['dbp_mae']:10.2f} "
              f"{results['sbp_rmse']:10.2f} {results['dbp_rmse']:10.2f}")
        
        total_samples += n
        weighted_sbp_mae += n * results['sbp_mae']
        weighted_dbp_mae += n * results['dbp_mae']
        weighted_sbp_rmse += n * results['sbp_rmse']
        weighted_dbp_rmse += n * results['dbp_rmse']
    
    print("-" * 80)
    print(f"{'Average':12} {total_samples:8d} "
          f"{weighted_sbp_mae/total_samples:10.2f} {weighted_dbp_mae/total_samples:10.2f} "
          f"{weighted_sbp_rmse/total_samples:10.2f} {weighted_dbp_rmse/total_samples:10.2f}")
    
    # 保存詳細結果
    np.save('personalized_evaluation_results.npy', all_results)
    print("\n詳細結果已保存至 personalized_evaluation_results.npy")

if __name__ == "__main__":
    main() 