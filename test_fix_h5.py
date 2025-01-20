import h5py
import numpy as np
from tqdm import tqdm

def compute_average_time_diff(seq1, seq2, fs=125):
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

def fix_pat_in_h5(h5_path):
    """
    在現有 h5 檔案裡，重新用 PPG_Turns 和 ECG_RealPeaks 計算 PAT，
    然後覆蓋到 vascular_properties[:,1]。
    """
    try:
        with h5py.File(h5_path, "r+") as f:
            # 取得 dataset
            annotations = f["annotations"]
            vascular = f["vascular_properties"]
            
            n_samples = annotations.shape[0]
            print(f"[INFO] 修正 {h5_path}, 共 {n_samples} 筆")
            
            for i in tqdm(range(n_samples)):
                anno = annotations[i]   # shape = (1250,4)
                # 取出 PPG_Turns (col=2) 與 ECG_RealPeaks (col=0) 的位置索引
                ppg_turns = np.where(anno[:,2] == 1)[0]
                ecg_peaks = np.where(anno[:,0] == 1)[0]
                
                # 計算新的 PAT
                pat_val = compute_average_time_diff(ppg_turns, ecg_peaks, fs=125)
                
                # 覆蓋 vascular_properties[i,1]
                # 先把原值取出(可能有 ptt_val)
                old_vals = vascular[i]          # shape=(2,) => [ptt_val, pat_val]
                old_vals[1] = pat_val           # 只更新第二個欄位
                vascular[i] = old_vals          # 寫回
            
            print(f"[INFO] 修正完畢: {h5_path}")
    except OSError as e:
        print(f"[ERROR] 無法開啟檔案 {h5_path}: {str(e)}")
        print("請確保沒有其他程式正在使用此檔案")
    except Exception as e:
        print(f"[ERROR] 處理檔案時發生錯誤 {h5_path}: {str(e)}")

if __name__ == "__main__":
    h5_files = [
        "training_data_VitalDB_quality/training_1.h5",
        "training_data_VitalDB_quality/training_2.h5",
        "training_data_VitalDB_quality/training_3.h5",
        "training_data_VitalDB_quality/training_4.h5",
        "training_data_VitalDB_quality/training_5.h5",

        "training_data_VitalDB_quality/validation.h5",
        # "training_data_VitalDB_quality/test.h5"
    ]
    for fp in h5_files:
        fix_pat_in_h5(fp)
