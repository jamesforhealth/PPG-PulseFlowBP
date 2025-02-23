import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_rr_intervals(annotations, fs=125):
    """
    計算 ECG R-peaks 之間的平均時間間距
    annotations: shape (1250, 4), 其中第一列是 ECG_RealPeaks
    returns: 平均 RR interval (單位: ms)
    """
    r_peaks = np.where(annotations[:, 0] == 1)[0]
    if len(r_peaks) < 2:
        return 0.0
    
    intervals = np.diff(r_peaks)
    mean_interval_samples = np.mean(intervals)
    mean_interval_ms = (mean_interval_samples / fs) * 1000.0
    return mean_interval_ms

def update_h5_with_rr_intervals(h5_path):
    """
    更新單一 h5 檔案，添加 RR intervals 作為第三個 vascular property
    """
    temp_path = h5_path.parent / f"temp_{h5_path.name}"
    
    # 1. 讀取原始檔案並計算新的 RR intervals
    with h5py.File(h5_path, 'r') as f_in:
        n_samples = len(f_in['annotations'])
        rr_intervals = np.zeros(n_samples, dtype=np.float32)
        
        # 計算每個 segment 的 RR intervals
        for i in range(n_samples):
            annotations = f_in['annotations'][i]
            rr_intervals[i] = compute_rr_intervals(annotations)
        
        # 建立新的檔案並複製所有資料
        with h5py.File(temp_path, 'w') as f_out:
            # 複製所有原始 datasets
            for key in f_in.keys():
                if key != 'vascular_properties':
                    f_in.copy(key, f_out)
            
            # 建立新的 vascular_properties，包含第三個特徵
            old_vp = f_in['vascular_properties'][:]
            new_vp = np.zeros((n_samples, 3), dtype=np.float32)
            new_vp[:, :2] = old_vp  # 複製原有的兩個特徵
            new_vp[:, 2] = rr_intervals  # 加入新的 RR intervals
            
            f_out.create_dataset('vascular_properties', data=new_vp)
    
    # 2. 替換原始檔案
    h5_path.unlink()  # 刪除原始檔案
    temp_path.rename(h5_path)  # 將臨時檔案重命名為原始檔名

def main():
    # 設定目錄
    data_dir = Path("training_data_VitalDB_quality")
    
    # 蒐集所有需要處理的 h5 檔案
    h5_files = []
    h5_files.extend(data_dir.glob("training_*.h5"))
    h5_files.extend(data_dir.glob("validation.h5"))
    h5_files.extend(data_dir.glob("test.h5"))
    
    print(f"找到 {len(h5_files)} 個 h5 檔案需要更新")
    
    # 逐一處理每個檔案
    for h5_path in tqdm(h5_files, desc="Processing H5 files"):
        try:
            update_h5_with_rr_intervals(h5_path)
            print(f"Successfully updated {h5_path.name}")
        except Exception as e:
            print(f"Error processing {h5_path.name}: {e}")

if __name__ == "__main__":
    main()