import os
import h5py
import numpy as np
from tqdm import tqdm

def find_real_peaks(signal, original_peaks, window_size=5):
    """
    從original_peaks位置開始，根據信號變化方向尋找最近的局部極大值
    """
    real_peaks = []
    signal_len = len(signal)
    
    for peak in original_peaks:
        if peak >= signal_len:
            continue
            
        # 檢查是否已經在局部極大值
        if peak > 0 and peak < signal_len-1:
            if signal[peak] >= signal[peak-1] and signal[peak] >= signal[peak+1]:
                real_peaks.append(peak)  # 已經在極大值，不需要移動
                continue
                
        # 檢查是否在極小值附近（需要在窗口內尋找最大值）
        if peak > 0 and peak < signal_len-1:
            if signal[peak] <= signal[peak-1] and signal[peak] <= signal[peak+1]:
                # 在極小值，使用窗口搜尋
                start = max(0, peak - window_size)
                end = min(signal_len, peak + window_size + 1)
                window = signal[start:end]
                local_max_idx = start + np.argmax(window)
                real_peaks.append(local_max_idx)
                continue
        
        # 其他情況：根據斜率方向尋找最近的極大值
        left_idx = peak
        right_idx = peak
        
        # 向左搜尋
        while left_idx > 0:
            if signal[left_idx-1] > signal[left_idx]:
                left_idx -= 1
            else:
                break
                
        # 向右搜尋
        while right_idx < signal_len-1:
            if signal[right_idx+1] > signal[right_idx]:
                right_idx += 1
            else:
                break
                
        # 比較左右兩側找到的點，選擇信號值較大的
        if signal[left_idx] > signal[right_idx]:
            real_peaks.append(left_idx)
        else:
            real_peaks.append(right_idx)
    
    return np.array(real_peaks)

def process_mimic_files(folder_path):
    """處理MIMIC資料集中的所有檔案，找出真實的ECG peaks並保存"""
    # 獲取所有.mat檔案
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    
    # 使用tqdm顯示進度
    for filename in tqdm(mat_files, desc="處理檔案"):
        file_path = os.path.join(folder_path, filename)
        print(f"\n處理檔案: {filename}")
        
        try:
            with h5py.File(file_path, 'r+') as f:
                matdata = f['Subj_Wins']
                
                # 刪除舊的資料集（如果存在）
                if 'ECG_RealPeaks' in matdata:
                    print(f"刪除舊的 ECG_RealPeaks 資料集")
                    del matdata['ECG_RealPeaks']
                
                if 'ECG_RealPeaks_Refs' in f:
                    del f['ECG_RealPeaks_Refs']
                
                # 刪除舊的RealPeaks片段
                for key in list(f.keys()):
                    if key.startswith('RealPeaks_'):
                        del f[key]
                
                # 取得所需的資料
                ecg_f_refs = matdata['ECG_F'][0]
                ecg_peaks_refs = matdata['ECG_RPeaks'][0]
                n_segments = len(ecg_f_refs)
                
                # 創建新的資料集來存儲真實peaks
                real_peaks_data = []
                
                # 處理每個片段
                for i in tqdm(range(n_segments), desc="處理片段", leave=False):
                    # 讀取當前片段的資料
                    ecg_f = f[ecg_f_refs[i]][:].flatten()
                    original_peaks = f[ecg_peaks_refs[i]][:].flatten().astype(int)
                    
                    # 找出真實peaks
                    real_peaks = find_real_peaks(ecg_f, original_peaks)
                    
                    # 創建新的資料集來存儲這個片段的真實peaks
                    real_peaks_ds = f.create_dataset(
                        f'/RealPeaks_{i}', 
                        data=real_peaks.reshape(1, -1),  # 確保形狀是 (1, N)
                        dtype=np.int32
                    )
                    real_peaks_data.append(real_peaks_ds.ref)
                
                # 創建新的資料集來存儲所有片段的參考
                real_peaks_refs = f.create_dataset(
                    '/ECG_RealPeaks_Refs',
                    shape=(1, len(real_peaks_data)),
                    data=real_peaks_data,
                    dtype=h5py.special_dtype(ref=h5py.Reference)
                )
                
                # 將新的資料集加入到Subj_Wins
                matdata.create_dataset(
                    'ECG_RealPeaks',
                    shape=(1, 1),
                    data=[real_peaks_refs.ref],
                    dtype=h5py.special_dtype(ref=h5py.Reference)
                )
                
                print(f"檔案 {filename} 處理完成")
                
        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {str(e)}")
            continue

def main():
    # 設定MIMIC資料集的路徑
    mimic_folder = "D:\\PulseDB\\PulseDB_MIMIC"
    
    # 確認資料夾存在
    if not os.path.exists(mimic_folder):
        print(f"錯誤: 找不到資料夾 {mimic_folder}")
        return
    
    # 開始處理檔案
    print(f"開始處理MIMIC資料集...")
    process_mimic_files(mimic_folder)
    print("\n所有檔案處理完成！")

if __name__ == "__main__":
    main() 