import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_first_derivative(signal):
    return np.gradient(signal)

def calculate_second_derivative(signal):
    return np.gradient(np.gradient(signal))

def encode_personal_info(person_dict):
    """
    編碼個人資訊為5維向量 (Age, BMI, Gender, Height, Weight)
    """
    info = np.zeros(5)
    info[0] = person_dict.get('Age', 0.0)
    info[1] = person_dict.get('BMI', 0.0)
    info[2] = person_dict.get('Gender', 0.0)
    info[3] = person_dict.get('Height', 0.0)
    info[4] = person_dict.get('Weight', 0.0)
    return info

class PersonalizedMIMICDataPreparator:
    """
    用於「針對每個 受試者檔/每個.mat」獨立做資料預處理，
    再存到 personalized_training_data_MIMIC/<filename>_train.h5, <filename>_val.h5
    """
    def __init__(self, data_dir="PulseDB_MIMIC", output_dir="personalized_training_data_MIMIC"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_file(self, file_path):
        """
        讀取單一受試者 .mat => 內有 PPG_Raw, ABP_Raw, ECG_Raw
        回傳 list of dict => [{'ppg','vpg','apg','segsbp','segdbp','personal_info'}, ... ]
        """
        try:
            with h5py.File(file_path, 'r') as f:
                if 'Subj_Wins' not in f:
                    print(f"Warning: 'Subj_Wins' not found in {file_path}")
                    return []
                
                matdata = f['Subj_Wins']
                
                # 檢查必要的數據欄位是否存在
                required_fields = ['PPG_Raw', 'ECG_Raw', 'ABP_Raw']
                for field in required_fields:
                    if field not in matdata:
                        print(f"Warning: '{field}' not found in {file_path}")
                        return []

                # 讀取波形數據
                ppg_raw = np.array([f[ref][:].flatten() for ref in matdata['PPG_Raw'][0]])
                ecg_raw = np.array([f[ref][:].flatten() for ref in matdata['ECG_Raw'][0]])
                abp_raw = np.array([f[ref][:].flatten() for ref in matdata['ABP_Raw'][0]])

                # 檢查數據形狀
                if len(ppg_raw) == 0 or len(ecg_raw) == 0 or len(abp_raw) == 0:
                    print(f"Warning: Empty data arrays in {file_path}")
                    return []

                # 假設所有 MIMIC 數據的個人資訊都是缺失的，使用預設值
                personal_info = {
                    'Age': 0.0,
                    'BMI': 0.0,
                    'Gender': 0.0,
                    'Height': 0.0,
                    'Weight': 0.0
                }
                personal_info_encoded = encode_personal_info(personal_info)

                processed_data = []
                
                # 處理每個片段
                for seg_idx in range(len(ppg_raw)):
                    ppg = ppg_raw[seg_idx]
                    ecg = ecg_raw[seg_idx]
                    abp = abp_raw[seg_idx]
                    
                    # 檢查信號長度
                    if len(ppg) != 1250 or len(ecg) != 1250 or len(abp) != 1250:
                        continue

                    # 從 ABP 波形中提取 SBP/DBP
                    sbp_val = float(np.max(abp))
                    dbp_val = float(np.min(abp))
                    
                    # 計算 VPG, APG
                    vpg = calculate_first_derivative(ppg)
                    apg = calculate_second_derivative(ppg)

                    data = {
                        'ppg': ppg,
                        'vpg': vpg,
                        'apg': apg,
                        'ecg': ecg,
                        'segsbp': sbp_val,
                        'segdbp': dbp_val,
                        'personal_info': personal_info_encoded
                    }
                    processed_data.append(data)

                return processed_data
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def save_data(self, data_list, out_path):
        """
        把 data_list 寫成 h5: ppg, vpg, apg, segsbp, segdbp, personal_info
        shape=(N,1250)
        """
        if len(data_list) == 0:
            print(f"[Skip] {out_path}, data_list=0.")
            return

        n_samples = len(data_list)
        input_len = 1250
        info_dim = len(data_list[0]['personal_info'])

        with h5py.File(out_path, 'w') as f_out:
            f_out.create_dataset('ppg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('vpg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('apg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('ecg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('segsbp', (n_samples,), dtype='float32')
            f_out.create_dataset('segdbp', (n_samples,), dtype='float32')
            f_out.create_dataset('personal_info', (n_samples, info_dim), dtype='float32')

            for i, dic in enumerate(data_list):
                f_out['ppg'][i] = dic['ppg']
                f_out['vpg'][i] = dic['vpg']
                f_out['apg'][i] = dic['apg']
                f_out['ecg'][i] = dic['ecg']
                f_out['segsbp'][i] = dic['segsbp']
                f_out['segdbp'][i] = dic['segdbp']
                f_out['personal_info'][i] = dic['personal_info']

    def create_personalized_data(self, file_path):
        """
        針對該 file_path(一個受試者 or 檔案),
        做預處理並拆成 80% train, 20% val => 
        輸出到: <filename>_train.h5, <filename>_val.h5
        """
        data_list = self.process_file(file_path)
        
        # 如果資料筆數少於10，則跳過
        if len(data_list) < 10:
            print(f"[Skip] {file_path} => segments < 10, skip output.")
            return
        
        # split
        split_idx = int(len(data_list) * 0.8)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]

        fname_stem = Path(file_path).stem  # ex: "p065484.mat" => "p065484"
        train_out = self.output_dir / f"{fname_stem}_train.h5"
        val_out = self.output_dir / f"{fname_stem}_val.h5"

        self.save_data(train_data, train_out)
        self.save_data(val_data, val_out)
        print(f"[Done] {file_path} => {train_out}, {val_out}")

if __name__ == '__main__':
    preparator = PersonalizedMIMICDataPreparator(
        data_dir="PulseDB_MIMIC",
        output_dir="personalized_training_data_MIMIC"
    )

    # 針對 data_dir 下所有 .mat 檔案，
    # 為每個檔做 create_personalized_data => 產生 train/val
    mat_files = list(preparator.data_dir.glob("*.mat"))
    for f in tqdm(mat_files, desc="Convert each person/file"):
        preparator.create_personalized_data(f)
    
    print("\n[All Done] 請查看 personalized_training_data_MIMIC/ 下的 *_train.h5, *_val.h5") 