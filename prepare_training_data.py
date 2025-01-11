import os
import h5py
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

##########################
# 1) 個人資訊編碼 (可自行調整)
##########################
def encode_personal_info(personal_info):
    """
    這裡只保留 Age, BMI, Height, Weight 四項，
    不再使用 gender 進行 one-hot。
    
    若您想保留 gender，可自行調整。
    """
    # 依照順序固定為 [Age, BMI, Height, Weight]
    # 若對應到 f 檔案裡沒有(=0.0)，則記成 0.0
    encoded_info = []
    for key in ['Age','BMI','Gender','Height','Weight']:
        if key == 'Gender':
            val = personal_info.get(key, 0.0)
            if val == 77:
                encoded_info.append(1.0)
            elif val == 70:
                encoded_info.append(-1.0)
            else:
                print(f'Gender value: {val}')
                encoded_info.append(0.0)
        else:
            val = personal_info.get(key, 0.0)
            try:
                encoded_info.append(float(val))
            except ValueError:
                encoded_info.append(0.0)
    return encoded_info

##########################
# 2) 差分計算函式
##########################
def calculate_first_derivative(sig_1d):
    """
    一階導數 VPG:
      vpg[i] = (sig_1d[i+1] - sig_1d[i-1]) / 2
    首尾做邊界處理
    sig_1d: shape=(1024,)
    """
    vpg = np.zeros_like(sig_1d)
    # 中間
    vpg[1:-1] = (sig_1d[2:] - sig_1d[:-2]) / 2
    # 邊界
    vpg[0] = vpg[1]
    vpg[-1] = vpg[-2]
    return vpg

def calculate_second_derivative(sig_1d):
    """
    二階導數 APG:
      apg[i] = sig_1d[i+1] - 2*sig_1d[i] + sig_1d[i-1]
    首尾做邊界處理
    sig_1d: shape=(1024,)
    """
    apg = np.zeros_like(sig_1d)
    apg[1:-1] = sig_1d[2:] - 2*sig_1d[1:-1] + sig_1d[:-2]
    apg[0] = apg[1]
    apg[-1] = apg[-2]
    return apg

##########################
# 3) DatasetPreparator
##########################
class DatasetPreparator:
    """
    - 讀取 processed_data/ 下的 *.mat.h5 檔
    - 將每個檔案的每個 segment (1250 點) 切成 2 筆資料:
        front: 前 1024 點
        back : 後 1024 點
      並各自計算 VPG, APG
      同時只保留 [Age, BMI, Height, Weight] 4 維個人資訊
      + segsbp, segdbp 當標籤
    - 分成 10 fold，其中 1 fold 做 val/test，其餘 9 fold 各自存成 training_1~9.h5
    """

    def __init__(self, data_dir="processed_data", output_dir="training_data", n_folds=10):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.output_dir.mkdir(exist_ok=True)

    def calculate_bp_values(self, abp_segment):
        """
        計算該 1024 點 ABP 的 SBP, DBP
        若超過合理範圍就當作 None 篩掉
        """
        sbp = np.max(abp_segment)
        dbp = np.min(abp_segment)
        if sbp > 300 or sbp < 20 or dbp > 300 or dbp < 20:
            return None
        return sbp, dbp
    
    def process_file_1250(self, file_path):
        """
        讀取單個 .h5 檔，並回傳多筆資料
        每筆包含:
          ppg, vpg, apg, ecg (各 1250 點)
          segsbp, segdbp
          personal_info(4 維)
        """
        with h5py.File(file_path, 'r') as f:
            # 1) 讀個人資訊 (只取 Age, BMI, Height, Weight)
            personal_info_keys = ['Age','BMI','Gender','Height','Weight']
            person_dict = {}
            for key in personal_info_keys:
                if key in f:
                    val_arr = f[key][:]
                    if len(val_arr.shape) > 0:
                        person_dict[key] = val_arr.flatten()[0]
                    else:
                        person_dict[key] = float(val_arr)
                else:
                    # 若檔內沒有，就給 0.0
                    person_dict[key] = 0.0
            personal_info_encoded = encode_personal_info(person_dict)

            # 2) 讀 raw 波形
            ppg_raw = f['PPG_Raw'][:]     # (n_segments,1250)
            ecg_raw = f['ECG_Raw'][:]
            abp_raw = f['ABP_Raw'][:]
            segsbp = f['SegSBP'][:]       # (n_segments,1) or (n_segments,1,1)
            segdbp = f['SegDBP'][:]

            processed_data = []

            # 3) 逐段處理
            for seg_idx in range(len(ppg_raw)):
                # segsbp, segdbp 取 [0] => 取出 scalar
                sbp_val = segsbp[seg_idx][0]
                dbp_val = segdbp[seg_idx][0]

                # 使用整個 segment (1250 點)
                ppg = ppg_raw[seg_idx]
                ecg = ecg_raw[seg_idx]
                abp = abp_raw[seg_idx]

                # 檢查 ABP 的血壓值是否合理
                bp = self.calculate_bp_values(abp)
                if bp is not None:
                    # 計算 VPG, APG
                    vpg = calculate_first_derivative(ppg)
                    apg = calculate_second_derivative(ppg)

                    data = {
                        'ppg': ppg,       # (1250,)
                        'vpg': vpg,
                        'apg': apg,
                        'ecg': ecg,
                        'segsbp': sbp_val,      # 直接用 segsbp
                        'segdbp': dbp_val,
                        'personal_info': personal_info_encoded  # (4,)
                    }
                    processed_data.append(data)

            return processed_data    
    def process_file(self, file_path):
        """
        讀取單個 .h5 檔，並回傳多筆 (front+back) 資料
        每筆包含:
          ppg, vpg, apg, ecg (各 1024 點)
          segsbp, segdbp
          personal_info(4 維)
        """
        with h5py.File(file_path, 'r') as f:
            # 1) 讀個人資訊 (只取 Age, BMI, Height, Weight)
            personal_info_keys = ['Age','BMI','Gender','Height','Weight']
            person_dict = {}
            for key in personal_info_keys:
                if key in f:
                    val_arr = f[key][:]
                    if len(val_arr.shape) > 0:
                        person_dict[key] = val_arr.flatten()[0]
                    else:
                        person_dict[key] = float(val_arr)
                else:
                    # 若檔內沒有，就給 0.0
                    person_dict[key] = 0.0
            personal_info_encoded = encode_personal_info(person_dict)

            # 2) 讀 raw 波形
            ppg_raw = f['PPG_Raw'][:]     # (n_segments,1250)
            ecg_raw = f['ECG_Raw'][:]
            abp_raw = f['ABP_Raw'][:]
            segsbp = f['SegSBP'][:]       # (n_segments,1) or (n_segments,1,1)
            segdbp = f['SegDBP'][:]

            processed_data = []

            # 3) 逐段處理
            for seg_idx in range(len(ppg_raw)):
                # 3-1) segsbp, segdbp 取 [0] => 取出 scalar
                sbp_val = segsbp[seg_idx][0]
                dbp_val = segdbp[seg_idx][0]

                # ========== front (0~1023) ==========
                ppg_front = ppg_raw[seg_idx][:1024]
                ecg_front = ecg_raw[seg_idx][:1024]
                abp_front = abp_raw[seg_idx][:1024]  # 用來檢查SBP/DBP區間

                # 檢查 front 血壓值是否合理
                bp_front = self.calculate_bp_values(abp_front)
                if bp_front is not None: 
                    # 計算前1024的 vpg, apg
                    vpg_front = calculate_first_derivative(ppg_front)
                    apg_front = calculate_second_derivative(ppg_front)

                    front_data = {
                        'ppg': ppg_front,       # (1024,)
                        'vpg': vpg_front,
                        'apg': apg_front,
                        'ecg': ecg_front,
                        'segsbp': sbp_val,      # 直接用 segsbp
                        'segdbp': dbp_val,
                        'personal_info': personal_info_encoded  # (4,)
                    }
                    processed_data.append(front_data)

                # ========== back (1250-1024=226 ~ 1249) ==========
                ppg_back = ppg_raw[seg_idx][-1024:]
                ecg_back = ecg_raw[seg_idx][-1024:]
                abp_back = abp_raw[seg_idx][-1024:]

                bp_back = self.calculate_bp_values(abp_back)
                if bp_back is not None:
                    vpg_back = calculate_first_derivative(ppg_back)
                    apg_back = calculate_second_derivative(ppg_back)

                    back_data = {
                        'ppg': ppg_back,
                        'vpg': vpg_back,
                        'apg': apg_back,
                        'ecg': ecg_back,
                        'segsbp': sbp_val,
                        'segdbp': dbp_val,
                        'personal_info': personal_info_encoded
                    }
                    processed_data.append(back_data)
            
            return processed_data
    
    def write_split_h5(self, output_path, file_list):
        """
        將 file_list 的所有資料統合, 寫成 output_path(.h5)
        欄位:
          ppg, vpg, apg, ecg : (N, 1024)
          segsbp, segdbp     : (N,)
          personal_info      : (N,4)
        """
        print(f"\n[INFO] 正在寫入 {output_path} | 檔案數量: {len(file_list)}")
        all_data = []
        
        # 逐檔處理
        for file_path in tqdm(file_list):
            data_list = self.process_file_1250(file_path)
            all_data.extend(data_list)
        
        print(f"  => {output_path} 資料筆數: {len(all_data)}")
        if len(all_data) == 0:
            print(f"  => 警告: {output_path} 無有效資料，略過建立!")
            return
        
        # personal_info 長度 (4)
        info_dim = len(all_data[0]['personal_info'])
        input_len = 1250#1024
        # 建立 HDF5
        with h5py.File(output_path, 'w') as f_out:
            n_samples = len(all_data)
            # 建立對應 dataset
            f_out.create_dataset('ppg',     (n_samples, input_len), dtype='float32')
            f_out.create_dataset('vpg',     (n_samples, input_len), dtype='float32')
            f_out.create_dataset('apg',     (n_samples, input_len), dtype='float32')
            f_out.create_dataset('ecg',     (n_samples, input_len), dtype='float32')
            f_out.create_dataset('segsbp',  (n_samples,),      dtype='float32')
            f_out.create_dataset('segdbp',  (n_samples,),      dtype='float32')
            f_out.create_dataset('personal_info', (n_samples, info_dim), dtype='float32')

            # 寫入
            for i, item in enumerate(all_data):
                f_out['ppg'][i]    = item['ppg']
                f_out['vpg'][i]    = item['vpg']
                f_out['apg'][i]    = item['apg']
                f_out['ecg'][i]    = item['ecg']
                f_out['segsbp'][i] = item['segsbp']
                f_out['segdbp'][i] = item['segdbp']
                f_out['personal_info'][i] = item['personal_info']
    
    def create_folds(self):
        """
        1) 收集所有 *.mat.h5 檔 -> 隨機打亂 -> 分成10等份 fold
        2) 指定某個fold(此例 fold0)對半分 val/test
        3) 其餘9 fold => training_1.h5 ~ training_9.h5
        """
        files = list(self.data_dir.glob('*.mat.h5'))
        total_files = len(files)
        np.random.shuffle(files)
        
        print(f"[INFO] 總共有 {total_files} 個檔案.")
        
        fold_size = total_files // self.n_folds
        folds = []
        
        # 建立10個fold
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            if i == self.n_folds - 1:  # 最後一fold把剩餘檔案都拿完
                end_idx = total_files
            fold_files = files[start_idx:end_idx]
            folds.append(fold_files)
            print(f"fold {i}: {len(fold_files)} files")
        
        # 指定 fold_idx_for_val_test = 0
        fold_idx_for_val_test = 0
        val_test_files = folds[fold_idx_for_val_test]
        
        # 對半分成 val / test
        half = len(val_test_files) // 2
        val_files  = val_test_files[:half]
        test_files = val_test_files[half:]
        print(f"[INFO] fold {fold_idx_for_val_test} => validation: {len(val_files)}, test: {len(test_files)}")
        
        # 其餘9 folds => training_1~training_9.h5
        train_fold_count = 1
        for i in range(self.n_folds):
            if i == fold_idx_for_val_test:
                continue
            out_path = self.output_dir / f"training_{train_fold_count}.h5"
            self.write_split_h5(out_path, folds[i])
            train_fold_count += 1
        
        # 寫出 validation.h5 / test.h5
        val_path  = self.output_dir / "validation.h5"
        test_path = self.output_dir / "test.h5"
        self.write_split_h5(val_path, val_files)
        self.write_split_h5(test_path, test_files)
        
        # (可選) 輸出檔案列表
        with open(self.output_dir / "val_files.txt", "w", encoding="utf-8") as f_list:
            for fp in val_files:
                f_list.write(f"{fp.name}\n")
        with open(self.output_dir / "test_files.txt", "w", encoding="utf-8") as f_list:
            for fp in test_files:
                f_list.write(f"{fp.name}\n")
        
        train_fold_count = 1
        for i in range(self.n_folds):
            if i == fold_idx_for_val_test:
                continue
            with open(self.output_dir / f"training_{train_fold_count}_files.txt","w",encoding="utf-8") as f_list:
                for fp in folds[i]:
                    f_list.write(f"{fp.name}\n")
            train_fold_count += 1
        
        print("\n[INFO] Done. 產生 9 個 training_x.h5, 1 個 validation.h5, 1 個 test.h5.")


if __name__ == '__main__':
    preparator = DatasetPreparator(
        data_dir="processed_data",
        output_dir="training_data_1250",
        n_folds=10
    )
    preparator.create_folds()
