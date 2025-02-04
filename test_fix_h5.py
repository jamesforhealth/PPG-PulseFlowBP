import h5py
import numpy as np
from tqdm import tqdm

def compute_average_time_diff(seq1, seq2, fs=125):
    """計算兩個序列間的平均時間差"""
    seq1 = np.sort(seq1)
    seq2 = np.sort(seq2)
    time_diffs_samples = []
    for idx1 in seq1:
        idx2_candidates = seq2[seq2 < idx1]
        if len(idx2_candidates) == 0:
            continue
        idx2 = idx2_candidates[-1]
        diff_samples = idx1 - idx2
        time_diffs_samples.append(diff_samples)
    if len(time_diffs_samples) == 0:
        return 0.0
    avg_diff_samples = np.mean(time_diffs_samples)
    return (avg_diff_samples / fs) * 1000.0

def normalize_signal(signal):
    """將信號正規化到 0-1 之間"""
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max == signal_min:
        return np.zeros_like(signal)
    return (signal - signal_min) / (signal_max - signal_min)

def compute_ppg_derivatives(ppg_signal):
    """計算 PPG 的一階和二階差分，並進行正規化"""
    # 一階差分
    first_diff = np.zeros_like(ppg_signal)
    first_diff[1:-1] = (ppg_signal[2:] - ppg_signal[:-2]) / 2
    first_diff[0] = ppg_signal[1] - ppg_signal[0]
    first_diff[-1] = ppg_signal[-1] - ppg_signal[-2]
    
    # 二階差分
    second_diff = np.zeros_like(ppg_signal)
    second_diff[2:-2] = (ppg_signal[4:] - 2*ppg_signal[2:-2] + ppg_signal[:-4]) / 4
    second_diff[0] = second_diff[2]
    second_diff[1] = second_diff[2]
    second_diff[-2] = second_diff[-3]
    second_diff[-1] = second_diff[-3]
    
    # 正規化
    first_diff_norm = normalize_signal(first_diff)
    second_diff_norm = normalize_signal(second_diff)
    
    return first_diff_norm, second_diff_norm

def process_h5_file(h5_path):
    """處理 h5 文件：添加正規化後的差分信號並更新 PAT"""
    try:
        with h5py.File(h5_path, "r+") as f:
            n_samples = f['ppg'].shape[0]
            print(f"[INFO] 處理 {h5_path}, 共 {n_samples} 筆")
            
            # 檢查是否已存在差分數據集
            if 'ppg_first_derivative' in f:
                del f['ppg_first_derivative']
            if 'ppg_second_derivative' in f:
                del f['ppg_second_derivative']
            
            # 創建新的數據集
            first_deriv_dataset = f.create_dataset('ppg_first_derivative', 
                                                 shape=f['ppg'].shape,
                                                 dtype=np.float32)
            second_deriv_dataset = f.create_dataset('ppg_second_derivative',
                                                  shape=f['ppg'].shape,
                                                  dtype=np.float32)
            
            # 處理每個片段
            for i in tqdm(range(n_samples)):
                # 計算正規化後的差分
                ppg_signal = f['ppg'][i]
                first_diff_norm, second_diff_norm = compute_ppg_derivatives(ppg_signal)
                
                # 保存差分結果
                first_deriv_dataset[i] = first_diff_norm
                second_deriv_dataset[i] = second_diff_norm
                
                # 更新 PAT
                anno = f['annotations'][i]
                ppg_turns = np.where(anno[:,2] == 1)[0]
                ecg_peaks = np.where(anno[:,0] == 1)[0]
                pat_val = compute_average_time_diff(ppg_turns, ecg_peaks, fs=125)
                
                # 更新 vascular_properties
                old_vals = f['vascular_properties'][i]
                old_vals[1] = pat_val
                f['vascular_properties'][i] = old_vals
            
            print(f"[INFO] 完成處理: {h5_path}")
            print(f"      - 添加了正規化的 PPG 一階差分")
            print(f"      - 添加了正規化的 PPG 二階差分")
            print(f"      - 更新了 PAT 值")
            
    except OSError as e:
        print(f"[ERROR] 無法開啟檔案 {h5_path}: {str(e)}")
        print("請確保沒有其他程式正在使用此檔案")
    except Exception as e:
        print(f"[ERROR] 處理檔案時發生錯誤 {h5_path}: {str(e)}")

if __name__ == "__main__":
    h5_files = [
        # "training_data_VitalDB_quality/training_1.h5",
        # "training_data_VitalDB_quality/training_2.h5",
        # "training_data_VitalDB_quality/training_3.h5",
        # "training_data_VitalDB_quality/training_4.h5",
        # "training_data_VitalDB_quality/training_5.h5",
        # "training_data_VitalDB_quality/training_6.h5",
        # "training_data_VitalDB_quality/training_7.h5",
        "training_data_VitalDB_quality/training_8.h5",
        "training_data_VitalDB_quality/training_9.h5",
        # "training_data_VitalDB_quality/validation.h5",
        # "training_data_VitalDB_quality/test.h5"
    ]
    
    for fp in h5_files:
        process_h5_file(fp)
