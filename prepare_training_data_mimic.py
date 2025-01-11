import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_annotation_matrix(length, peaks_dict):
    """
    將不同類型的特徵點轉換為 one-hot 編碼矩陣(4類):
      0: normal(背景)
      1: ECG_RealPeak
      2: PPG_SPeak
      3: PPG_Turn

    輸出 shape = (length, 4)
    """
    annotation_matrix = np.zeros((length, 4), dtype=np.float32)
    # 先標記整段都是第0類(normal)
    annotation_matrix[:, 0] = 1.0

    # ECG_RealPeaks => 第1維
    if 'ECG_RealPeaks' in peaks_dict:
        for loc in peaks_dict['ECG_RealPeaks']:
            if 0 <= loc < length:
                annotation_matrix[loc, 0] = 0.0  # 移除 normal
                annotation_matrix[loc, 1] = 1.0  # ECG_RealPeak

    # PPG_SPeaks => 第2維
    if 'PPG_SPeaks' in peaks_dict:
        for loc in peaks_dict['PPG_SPeaks']:
            if 0 <= loc < length:
                annotation_matrix[loc, 0] = 0.0
                annotation_matrix[loc, 2] = 1.0

    # PPG_Turns => 第3維
    if 'PPG_Turns' in peaks_dict:
        for loc in peaks_dict['PPG_Turns']:
            if 0 <= loc < length:
                annotation_matrix[loc, 0] = 0.0
                annotation_matrix[loc, 3] = 1.0

    return annotation_matrix

class MIMICDatasetPreparator:
    def __init__(self, data_dir="PulseDB_MIMIC", output_dir="training_data_1250_MIMIC", n_folds=10):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_folds = n_folds

    def extract_peaks_list(self, f, ref_array, num_segments):
        """
        輔助函式:
          - 讀取 matdata['XXX'][0] 的 ref array
          - ref_array長度 可能 != num_segments
          - 實際可用長度 => usable_len = min(len(ref_array), num_segments)
          - 其餘段數補空peak
        回傳: List[np.ndarray(int)], len = num_segments
        """
        peaks_list = [np.array([], dtype=int) for _ in range(num_segments)]
        actual_len = len(ref_array)
        usable_len = min(actual_len, num_segments)

        for i in range(usable_len):
            ref = ref_array[i]
            try:
                data = f[ref][()]  # 讀取 ref 指向的 dataset
                if data.size > 0:
                    peaks_list[i] = data.flatten().astype(int)
            except:
                pass

        return peaks_list

    def extract_personal_info(self, f):
        """
        從 .mat 檔案中提取個人資訊
        假設格式: age, gender, height, weight, bmi
        如果沒有資訊，返回預設值
        """
        personal_info = np.zeros(5, dtype=np.float32)  # 預設值
        try:
            if 'SubjectInfo' in f:
                info = f['SubjectInfo']
                if 'Age' in info:
                    personal_info[0] = float(info['Age'][()])
                if 'Gender' in info:
                    personal_info[1] = float(info['Gender'][()])  # 0=female, 1=male
                if 'Height' in info:
                    personal_info[2] = float(info['Height'][()])
                if 'Weight' in info:
                    personal_info[3] = float(info['Weight'][()])
                if 'BMI' in info:
                    personal_info[4] = float(info['BMI'][()])
        except Exception as e:
            print(f"Warning: Error extracting personal info: {str(e)}")
        return personal_info

    def process_file_1250(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'Subj_Wins' not in f:
                    print(f"[Warning] {file_path.name} => 'Subj_Wins' not found, skip.")
                    return None

                matdata = f['Subj_Wins']
                
                # 讀取個人資訊
                personal_info = self.extract_personal_info(f)

                # 檢查必要欄位
                if 'PPG_F' not in matdata or 'ABP_Raw' not in matdata:
                    print(f"[Warning] {file_path.name} => PPG_F/ABP_Raw missing, skip.")
                    return None

                # 讀取 PPG_F
                ppg_f_list = []
                for ref in matdata['PPG_F'][0]:
                    try:
                        arr = f[ref][()]
                        ppg_f_list.append(arr.flatten())
                    except:
                        ppg_f_list.append(np.array([], dtype=float))

                # 讀取 ABP_Raw
                abp_list = []
                for ref in matdata['ABP_Raw'][0]:
                    try:
                        arr = f[ref][()]
                        abp_list.append(arr.flatten())
                    except:
                        abp_list.append(np.array([], dtype=float))

                # 統一長度
                num_segments = len(ppg_f_list)
                if len(abp_list) != num_segments:
                    min_len = min(num_segments, len(abp_list))
                    ppg_f_list = ppg_f_list[:min_len]
                    abp_list = abp_list[:min_len]
                    num_segments = min_len

                if num_segments == 0:
                    print(f"[Warn] {file_path.name} => no valid segments after alignment.")
                    return None

                # 讀取所有peaks
                ecg_realpeaks_arr = [np.array([], dtype=int) for _ in range(num_segments)]
                ppg_speaks_arr = [np.array([], dtype=int) for _ in range(num_segments)]
                ppg_turns_arr = [np.array([], dtype=int) for _ in range(num_segments)]

                if 'ECG_RealPeaks' in matdata:
                    ref_array = matdata['ECG_RealPeaks'][0]
                    ecg_realpeaks_arr = self.extract_peaks_list(f, ref_array, num_segments)

                if 'PPG_SPeaks' in matdata:
                    ref_array = matdata['PPG_SPeaks'][0]
                    ppg_speaks_arr = self.extract_peaks_list(f, ref_array, num_segments)

                if 'PPG_Turns' in matdata:
                    ref_array = matdata['PPG_Turns'][0]
                    ppg_turns_arr = self.extract_peaks_list(f, ref_array, num_segments)

                # 從 ABP 提取 SBP/DBP
                seg_sbp, seg_dbp = [], []
                for abp_seg in abp_list:
                    if len(abp_seg) == 1250:
                        sbp_val = float(np.max(abp_seg))
                        dbp_val = float(np.min(abp_seg))
                    else:
                        sbp_val, dbp_val = 0.0, 0.0
                    seg_sbp.append(sbp_val)
                    seg_dbp.append(dbp_val)

                processed_data = []
                for i in range(num_segments):
                    if len(ppg_f_list[i]) != 1250:
                        continue

                    peaks_dict = {
                        'ECG_RealPeaks': ecg_realpeaks_arr[i],
                        'PPG_SPeaks': ppg_speaks_arr[i],
                        'PPG_Turns': ppg_turns_arr[i]
                    }
                    # 使用完整的 annotation matrix (4類)
                    ann_matrix = create_annotation_matrix(1250, peaks_dict)

                    data = {
                        'ppg': ppg_f_list[i],       # shape=(1250,)
                        'annotations': ann_matrix,   # shape=(1250,4)
                        'segsbp': seg_sbp[i],
                        'segdbp': seg_dbp[i],
                        'personal_info': personal_info  # shape=(5,)
                    }
                    processed_data.append(data)

                if len(processed_data) == 0:
                    return None
                return processed_data

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def write_split_h5(self, output_path, file_list):
        print(f"\n[INFO] 正在寫入 {output_path} | 檔案數量: {len(file_list)}")
        all_data = []

        for file_path in tqdm(file_list):
            data_list = self.process_file_1250(file_path)
            if data_list:
                all_data.extend(data_list)

        if len(all_data) == 0:
            print(f"  => 警告: {output_path} 無有效資料，略過建立!")
            return

        with h5py.File(output_path, 'w') as f_out:
            n_samples = len(all_data)
            f_out.create_dataset('ppg',          (n_samples, 1250),    dtype='float32')
            f_out.create_dataset('annotations',  (n_samples, 1250, 4), dtype='float32')  # 修正為4類
            f_out.create_dataset('segsbp',       (n_samples,),         dtype='float32')
            f_out.create_dataset('segdbp',       (n_samples,),         dtype='float32')
            f_out.create_dataset('personal_info', (n_samples, 5),      dtype='float32')

            for i, dic in enumerate(all_data):
                f_out['ppg'][i]          = dic['ppg']
                f_out['annotations'][i]  = dic['annotations']
                f_out['segsbp'][i]       = dic['segsbp']
                f_out['segdbp'][i]       = dic['segdbp']
                f_out['personal_info'][i] = dic['personal_info']

    def create_folds(self):
        files = list(self.data_dir.glob('*.mat'))
        total_files = len(files)
        np.random.shuffle(files)

        print(f"[INFO] 總共有 {total_files} 個檔案.")
        if total_files==0:
            print("[警告] 找不到任何 .mat 檔，流程結束。")
            return

        fold_size = total_files // self.n_folds
        folds = []
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx   = start_idx + fold_size
            if i == self.n_folds-1:
                end_idx = total_files
            fold_files = files[start_idx:end_idx]
            folds.append(fold_files)
            print(f"fold {i}: {len(fold_files)} files")

        # 指定 fold0 => val/test
        fold_idx_for_val_test = 0
        val_test_files = folds[fold_idx_for_val_test]
        half = len(val_test_files)//2
        val_files  = val_test_files[:half]
        test_files = val_test_files[half:]
        print(f"[INFO] fold {fold_idx_for_val_test} => validation: {len(val_files)}, test: {len(test_files)}")

        train_fold_count = 1
        for i in range(self.n_folds):
            if i==fold_idx_for_val_test:
                continue
            out_path = self.output_dir/f"training_{train_fold_count}.h5"
            self.write_split_h5(out_path, folds[i])
            train_fold_count+=1

        # val/test
        val_path  = self.output_dir/"validation.h5"
        test_path = self.output_dir/"test.h5"
        self.write_split_h5(val_path, val_files)
        self.write_split_h5(test_path, test_files)
        print("\n[INFO] Done. 產生多個 training_x.h5, 以及 validation.h5, test.h5.")

if __name__ == '__main__':
    preparator = MIMICDatasetPreparator(
        data_dir="PulseDB_MIMIC",
        output_dir="training_data_1250_MIMIC",
        n_folds=10
    )
    preparator.create_folds()
