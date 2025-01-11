import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from prepare_training_data import calculate_first_derivative, calculate_second_derivative, encode_personal_info

class PersonalizedDatasetPreparator:
    """
    用於「針對每個 受試者檔/每個.mat.h5」獨立做資料預處理，
    再存到 personalized_training_data/<filename>_train.h5, <filename>_val.h5
    """
    def __init__(self, data_dir="processed_data", output_dir="personalized_training_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def calculate_bp_values(self, abp_segment):
        """
        計算該 1250點(或任何長度) ABP 的 SBP/DBP，
        若超出合理範圍則 return None。
        (此處您也可另外做 wave-based BP check)
        """
        sbp = np.max(abp_segment)
        dbp = np.min(abp_segment)
        if sbp>300 or sbp<20 or dbp>300 or dbp<20:
            return None
        return sbp, dbp

    def process_file(self, file_path):
        """
        讀取單一受試者 .mat.h5 => 內有 PPG_Raw, ABP_Raw, SegSBP, SegDBP, Age,BMI,...
        回傳 list of dict => [{'ppg','vpg','apg','segsbp','segdbp','personal_info'}, ... ]
        """
        file_path = Path(file_path)
        with h5py.File(file_path, 'r') as f:
            # A) 個人資訊
            personal_info_keys = ['Age','BMI','Gender','Height','Weight']
            person_dict = {}
            for key in personal_info_keys:
                if key in f:
                    val_arr = f[key][:]
                    val = float(val_arr.flatten()[0]) if val_arr.size>0 else 0.0
                    person_dict[key] = val
                else:
                    person_dict[key] = 0.0
            personal_info_encoded = encode_personal_info(person_dict)

            # B) 讀 raw wave
            if 'PPG_Raw' not in f or 'ABP_Raw' not in f:
                print(f"[Warn] {file_path} missing PPG_Raw/ABP_Raw, skip.")
                return []
            ppg_raw = f['PPG_Raw'][:]  # (n_segments,1250)
            ecg_raw = f['ECG_Raw'][:]
            abp_raw = f['ABP_Raw'][:]
            segsbp  = f['SegSBP'][:]
            segdbp  = f['SegDBP'][:]

            processed_data = []
            n_segments = len(ppg_raw)

            for seg_idx in range(n_segments):
                sbp_val = float(segsbp[seg_idx][0])
                dbp_val = float(segdbp[seg_idx][0])

                ppg_1d = ppg_raw[seg_idx]
                ecg_1d = ecg_raw[seg_idx]
                abp_1d = abp_raw[seg_idx]

                # 檢查 ABP
                bp_check = self.calculate_bp_values(abp_1d)
                if bp_check is None:
                    continue

                # vpg, apg
                vpg_1d = calculate_first_derivative(ppg_1d)
                apg_1d = calculate_second_derivative(ppg_1d)

                data_dict = {
                    'ppg': ppg_1d,
                    'vpg': vpg_1d,
                    'apg': apg_1d,
                    'segsbp': sbp_val,
                    'segdbp': dbp_val,
                    'personal_info': personal_info_encoded,
                    'ecg': ecg_1d
                }
                processed_data.append(data_dict)

            return processed_data

    def create_personalized_data(self, file_path):
        """
        針對該 file_path(一個受試者 or 檔案),
        做預處理並拆成 80% train, 20% val => 
        輸出到: <filename>_train.h5, <filename>_val.h5
        """
        data_list = self.process_file(file_path)
        if len(data_list)==0:
            print(f"[Warn] {file_path} => no valid segments, skip output.")
            return
        
        # split
        split_idx = int(len(data_list)*0.8)
        train_data = data_list[:split_idx]
        val_data   = data_list[split_idx:]

        fname_stem = Path(file_path).stem  # ex: "some_person_data.mat"
        # 可能會想把 ".mat"一起移除 => 也可 .stem.stem
        # 這裡簡單用 .stem 即 "some_person_data.mat"

        train_out = self.output_dir / f"{fname_stem}_train.h5"
        val_out   = self.output_dir / f"{fname_stem}_val.h5"

        self.save_data(train_data, train_out)
        self.save_data(val_data,   val_out)
        print(f"[Done] {file_path} => {train_out}, {val_out}")

    def save_data(self, data_list, out_path):
        """
        把 data_list 寫成 h5: ppg, vpg, apg, segsbp, segdbp, personal_info
        shape=(N,1250)
        """
        if len(data_list)==0:
            print(f"[Skip] {out_path}, data_list=0.")
            return

        n_samples = len(data_list)
        input_len = 1250
        info_dim = len(data_list[0]['personal_info'])

        with h5py.File(out_path, 'w') as f_out:
            f_out.create_dataset('ppg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('vpg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('apg', (n_samples, input_len), dtype='float32')
            f_out.create_dataset('segsbp', (n_samples,), dtype='float32')
            f_out.create_dataset('segdbp', (n_samples,), dtype='float32')
            f_out.create_dataset('personal_info', (n_samples, info_dim), dtype='float32')
            f_out.create_dataset('ecg', (n_samples, input_len), dtype='float32')

            for i, dic in enumerate(data_list):
                f_out['ppg'][i]       = dic['ppg']
                f_out['vpg'][i]       = dic['vpg']
                f_out['apg'][i]       = dic['apg']
                f_out['segsbp'][i]    = dic['segsbp']
                f_out['segdbp'][i]    = dic['segdbp']
                f_out['personal_info'][i] = dic['personal_info']
                f_out['ecg'][i]       = dic['ecg']

####################################
# main
####################################
if __name__ == '__main__':
    preparator = PersonalizedDatasetPreparator(
        data_dir="processed_data", 
        output_dir="personalized_training_data"
    )

    # 針對 data_dir 下所有 .mat.h5 檔案，
    # 為每個檔做 create_personalized_data => 產生 train/val
    mat_files = list(preparator.data_dir.glob("*.mat.h5"))
    for f in tqdm(mat_files, desc="Convert each person/file"):
        preparator.create_personalized_data(f)
    
    print("\n[All Done] 請查看 personalized_training_data/ 下的 *_train.h5, *_val.h5")