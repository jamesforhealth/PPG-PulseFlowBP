import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from scipy import signal
from scipy import stats

################################################
# 1) 各種檢查函式：原樣保留
################################################

def check_ecg_quality(ecg_f, r_peaks):
    max_distance_threshold = 375
    min_distance_threshold = 63
    extend_boundary = np.array([0, *r_peaks, 1249])
    diff = np.diff(extend_boundary)
    diff2 = diff[1:-1]
    ecg_anomaly_score = compute_ecg_dtw_score(ecg_f, r_peaks)
    if ecg_anomaly_score is not None:
        threshold = 3.0   # 這個閾值可自行調整
        cond_score = (ecg_anomaly_score < threshold)
        cond_peaks = (len(r_peaks) > 4)
        cond_rr_ok = (np.all(diff < max_distance_threshold) and np.all(diff2 > min_distance_threshold))
        return True if (cond_score and cond_peaks and cond_rr_ok) else False
    else:
        return False

def compute_ecg_dtw_score(ecg_f, r_peaks):
    fs = 125
    r_peaks = r_peaks[r_peaks < len(ecg_f)]  # 避免越界
    if len(r_peaks) < 3:
        return float('inf')
    beat_segments = []
    for i in range(len(r_peaks)-1):
        start_idx = r_peaks[i]
        end_idx   = r_peaks[i+1]
        if end_idx <= start_idx:
            continue
        beat = ecg_f[start_idx:end_idx]
        beat_segments.append(beat)
    distances = []
    for i in range(len(beat_segments)-1):
        dist = simple_dtw(beat_segments[i], beat_segments[i+1])
        distances.append(dist)
    if len(distances) == 0:
        return 0.0
    anomaly_score = float(np.mean(distances))
    return anomaly_score

def simple_dtw(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_mat = np.full((n+1, m+1), np.inf)
    dtw_mat[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_mat[i, j] = cost + min(
                dtw_mat[i-1, j],   # 上
                dtw_mat[i, j-1],   # 左
                dtw_mat[i-1, j-1]  # 斜上
            )
    return dtw_mat[n, m]

def create_annotation_matrix(length, peaks_dict):
    anno_matrix = np.zeros((length, 4), dtype=np.float32)
    keys = ['ECG_RealPeaks', 'PPG_SPeaks', 'PPG_Turns']
    for idx, key in enumerate(keys):
        raw_peaks = peaks_dict.get(key, [])
        peaks = np.array(raw_peaks, dtype=np.int64)
        valid_peaks = peaks[(peaks >= 0) & (peaks < length)]
        valid_peaks = valid_peaks.astype(np.int64)
        anno_matrix[valid_peaks, idx] = 1.0
    return anno_matrix

def find_real_peaks(sig, peaks, fs=125):
    real_peaks = []
    search_radius = int(0.1 * fs)
    for peak in peaks:
        start = max(0, peak - search_radius)
        end   = min(len(sig), peak + search_radius)
        segment = sig[start:end]
        if len(segment) == 0:
            continue
        rel_idx = np.argmax(segment)
        real_peak = start + rel_idx
        real_peaks.append(real_peak)
    return np.array(real_peaks, dtype=int)

def compute_average_time_diff(seq1, seq2, fs=125):
    seq1 = np.sort(np.array(seq1))
    seq2 = np.sort(np.array(seq2))
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
    avg_diff_ms = (avg_diff_samples / fs) * 1000.0
    return avg_diff_ms

def get_scalar_from_ref(f, ref):
    try:
        data = f[ref][()]
        return float(data[0][0])
    except:
        return 0.0

def check_ppg_quality(ppg_raw, fs=125, window_size=3):
    nyq = fs / 2
    b, a = signal.butter(3, [0.5/nyq, 8/nyq], btype='band')
    ppg_filtered = signal.filtfilt(b, a, ppg_raw)
    
    samples_per_window = window_size * fs
    n_windows = len(ppg_filtered) // samples_per_window
    ssqi_values = []
    for i in range(n_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        window_data = ppg_filtered[start:end]
        ssqi = stats.skew(window_data)
        ssqi_values.append(ssqi)
    if len(ssqi_values) == 0:
        mean_ssqi = -999.0
    else:
        mean_ssqi = np.mean(ssqi_values)
    
    # 根據經驗阈值:
    if mean_ssqi > 0.07:
        quality = 'G1'
    elif mean_ssqi > -0.1:
        quality = 'G2'
    else:
        quality = 'G3'
    return quality, mean_ssqi

def check_peaks_order(peaks_dict):
    keys = ['ECG_RealPeaks', 'PPG_SPeaks', 'PPG_Turns']
    sequence_length = len(keys)
    for key in keys:
        if len(peaks_dict[key]) == 0:
            return False
    for key in keys:
        intervals = np.diff(peaks_dict[key])
        if len(intervals) == 0:
            continue
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        if min_interval == 0:
            return False
        if max_interval > 2 * min_interval:
            return False

    all_peaks = []
    for idx, key in enumerate(keys):
        p_sorted = np.sort(peaks_dict[key])
        labels = np.full(len(p_sorted), idx)
        all_peaks.extend(zip(p_sorted, labels))
    all_peaks.sort(key=lambda x: x[0])
    labels_sequence = [item[1] for item in all_peaks]
    if len(labels_sequence) < sequence_length:
        return False
    fixed_pattern = tuple(labels_sequence[:sequence_length])
    idx = 0
    while idx <= len(labels_sequence) - sequence_length:
        window = labels_sequence[idx:idx+sequence_length]
        if tuple(window) != fixed_pattern:
            return False
        idx += sequence_length
    if idx < len(labels_sequence):
        remaining = labels_sequence[idx:]
        if remaining != list(fixed_pattern)[:len(remaining)]:
            return False
    return True

def compute_time_diff_statistics(seq1, seq2, fs=125):
    """
    同時計算 seq1 與 seq2 之間的相對時間差之平均與標準差（單位: 毫秒）。
    注意這裡的邏輯和原先 compute_average_time_diff 類似，只是改為回傳 (mean, std)。
    """
    seq1 = np.sort(np.array(seq1))
    seq2 = np.sort(np.array(seq2))
    time_diffs_samples = []
    for idx1 in seq1:
        idx2_candidates = seq2[seq2 < idx1]
        if len(idx2_candidates) == 0:
            continue
        idx2 = idx2_candidates[-1]
        diff_samples = idx1 - idx2
        time_diffs_samples.append(diff_samples)

    if len(time_diffs_samples) == 0:
        return 0.0, 0.0

    time_diffs_samples = np.array(time_diffs_samples)
    mean_diff_samples = np.mean(time_diffs_samples)
    std_diff_samples = np.std(time_diffs_samples)
    mean_diff_ms = (mean_diff_samples / fs) * 1000.0
    std_diff_ms = (std_diff_samples / fs) * 1000.0
    return mean_diff_ms, std_diff_ms

def compute_rr_statistics(r_peaks, fs=125):
    """
    計算 ECG R-peaks 之間 RR interval 的平均和標準差（單位: 毫秒）。
    """
    r_peaks = np.sort(np.array(r_peaks))
    if len(r_peaks) < 2:
        return 0.0, 0.0
    
    intervals = np.diff(r_peaks)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    mean_interval_ms = (mean_interval / fs) * 1000.0
    std_interval_ms = (std_interval / fs) * 1000.0
    return mean_interval_ms, std_interval_ms

################################################
# 2) 資料處理主類別 (保留原先 process_file_1250 等邏輯)
################################################
class VitalDBDatasetPreparator:
    def __init__(self, data_dir="PulseDB_Vital", output_dir="training_data_VitalDB", n_folds=10):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_folds = n_folds

        # 這三個變數僅在 "原本" create_folds() 中統計用
        self.total_segments   = 0
        self.total_qualified  = 0
        self.total_discarded  = 0

    def extract_peaks(self, file_handle, ref):
        data = file_handle[ref][()]
        if isinstance(data, np.ndarray):
            if data.dtype == 'object':
                peaks = []
                for sub_ref in data:
                    sub_data = file_handle[sub_ref][()]
                    peaks.append(sub_data.flatten())
                if peaks:
                    peaks = np.concatenate(peaks).astype(int)
                else:
                    peaks = np.array([], dtype=int)
            else:
                peaks = data.flatten().astype(int)
        else:
            peaks = np.array([], dtype=int)
        return peaks

    def process_file_1250(self, file_path):
        """
        讀取單一 .mat 檔 => 逐 segment 檢查 => 回傳合格的 data_item list + 統計
        """
        try:
            with h5py.File(file_path, 'r') as f:
                matdata = f['Subj_Wins']

                # 1) 讀 ppg, ecg, abp
                ppg_f = []
                ecg_f = []
                abp_raw = []
                for ref in matdata['PPG_F'][0]:
                    ppg_f.append(f[ref][()].flatten())
                for ref in matdata['ECG_F'][0]:
                    ecg_f.append(f[ref][()].flatten())
                for ref in matdata['ABP_Raw'][0]:
                    abp_raw.append(f[ref][()].flatten())
                ppg_f = np.array(ppg_f)
                ecg_f = np.array(ecg_f)
                abp_raw = np.array(abp_raw)

                # 2) ECG_RPeaks => find_real_peaks => ECG_RealPeaks
                ecg_realpeaks_all = []
                for idx, ref in enumerate(matdata['ECG_RPeaks'][0]):
                    approx_peaks = self.extract_peaks(f, ref)
                    real_peaks = find_real_peaks(ecg_f[idx], approx_peaks, fs=125)
                    ecg_realpeaks_all.append(real_peaks)
                ecg_realpeaks_all = np.array(ecg_realpeaks_all, dtype=object)

                # 3) PPG_SPeaks, PPG_Turns
                ppg_speaks_all = []
                for ref in matdata['PPG_SPeaks'][0]:
                    peaks_ppg_spk = self.extract_peaks(f, ref)
                    ppg_speaks_all.append(peaks_ppg_spk)
                ppg_speaks_all = np.array(ppg_speaks_all, dtype=object)

                ppg_turns_all = []
                for ref in matdata['PPG_Turns'][0]:
                    peaks_ppg_tn = self.extract_peaks(f, ref)
                    ppg_turns_all.append(peaks_ppg_tn)
                ppg_turns_all = np.array(ppg_turns_all, dtype=object)

                # 4) SBP, DBP
                seg_sbp = []
                seg_dbp = []
                for abp_seg in abp_raw:
                    seg_sbp.append(np.max(abp_seg))
                    seg_dbp.append(np.min(abp_seg))
                seg_sbp = np.array(seg_sbp)
                seg_dbp = np.array(seg_dbp)

                # 5) 個人 info
                age_ref    = matdata['Age'][0][0]
                gender_ref = matdata['Gender'][0][0]
                weight_ref = matdata['Weight'][0][0]
                height_ref = matdata['Height'][0][0]
                age    = get_scalar_from_ref(f, age_ref)
                gender = get_scalar_from_ref(f, gender_ref)
                weight = get_scalar_from_ref(f, weight_ref)
                height = get_scalar_from_ref(f, height_ref)
                personal_info = np.array([age, gender, weight, height], dtype=np.float32)

                # 6) 逐 segment => annotation matrix + check
                processed_data= []
                n_segments= len(ppg_f)
                qualified=0
                discarded=0
                for i in range(n_segments):
                    # a) ECG
                    if not check_ecg_quality(ecg_f[i], ecg_realpeaks_all[i]):
                        discarded+=1
                        continue
                    # b) PPG
                    quality_label, ssqi_value = check_ppg_quality(ppg_f[i], fs=125)
                    if quality_label == 'G3':
                        discarded+=1
                        continue
                    # c) peaks order
                    peaks_dict= {
                        'ECG_RealPeaks': ecg_realpeaks_all[i],
                        'PPG_SPeaks': ppg_speaks_all[i],
                        'PPG_Turns': ppg_turns_all[i]
                    }
                    if not check_peaks_order(peaks_dict):
                        discarded+=1
                        continue

                    # d) 若通過 => annotation matrix / ptt(平均+std) / pat(平均+std) / rr(平均+std)
                    ptt_mean, ptt_std = compute_time_diff_statistics(
                        ppg_speaks_all[i], 
                        ppg_turns_all[i], 
                        fs=125
                    )
                    pat_mean, pat_std = compute_time_diff_statistics(
                        ppg_turns_all[i], 
                        ecg_realpeaks_all[i], 
                        fs=125
                    )
                    rr_mean, rr_std = compute_rr_statistics(ecg_realpeaks_all[i], fs=125)

                    annotation_mat= create_annotation_matrix(1250, peaks_dict)

                    data_item= {
                        'ppg': ppg_f[i],
                        'ecg': ecg_f[i],
                        'abp': abp_raw[i],
                        'annotations': annotation_mat,
                        'segsbp': seg_sbp[i],
                        'segdbp': seg_dbp[i],
                        'personal_info': personal_info,
                        'vascular_properties': np.array([ptt_mean, ptt_std, pat_mean, pat_std, rr_mean, rr_std], dtype=np.float32)
                    }
                    processed_data.append(data_item)
                    qualified+=1

                print(f"[{file_path.name}] total={n_segments}, ok={qualified}, discard={discarded}")
                return processed_data, n_segments, qualified, discarded

        except Exception as e:
            print(f"[Error] processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, 0, 0

    ################################################
    # A) 先「只產生」各 fold 的檔案列表 TXT
    ################################################
    def create_fold_lists(self):
        files = list(self.data_dir.glob("*.mat"))
        total_files = len(files)
        np.random.shuffle(files)
        print(f"[INFO] 总共有 {total_files} 个 .mat 档.")
        
        fold_size = total_files // self.n_folds
        folds = []
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size
            if i == self.n_folds - 1:
                end = total_files
            fold_files = files[start:end]
            folds.append(fold_files)
            print(f"fold{i}: {len(fold_files)} files")
        
        # 假設 fold 0 => val, test；其餘 => training
        fold_idx_for_val_test = 0
        val_test_files = folds[fold_idx_for_val_test]
        half = len(val_test_files)//2
        val_files = val_test_files[:half]
        test_files = val_test_files[half:]
        print(f"[Info] fold {fold_idx_for_val_test} => val={len(val_files)}, test={len(test_files)}")
        
        # 寫入文字檔
        # training_x_files.txt
        train_fold_count=1
        for i in range(self.n_folds):
            if i == fold_idx_for_val_test:
                continue
            out_txt = self.output_dir / f"training_{train_fold_count}_files.txt"
            with open(out_txt, "w", encoding="utf-8") as f_txt:
                for x in folds[i]:
                    f_txt.write(str(x.resolve()) + "\n")
            train_fold_count += 1

        # val_files.txt
        val_txt = self.output_dir / "val_files.txt"
        with open(val_txt, "w", encoding="utf-8") as f_list:
            for x in val_files:
                f_list.write(str(x.resolve()) + "\n")

        # test_files.txt
        test_txt = self.output_dir / "test_files.txt"
        with open(test_txt, "w", encoding="utf-8") as f_list:
            for x in test_files:
                f_list.write(str(x.resolve()) + "\n")
        
        print("\n[INFO] Folds檔案列表已產生於:", self.output_dir.resolve())

    ################################################
    # B) 根據「某一個檔案列表TXT」建立 HDF5
    ################################################
    def write_h5_from_file_list(self, file_list_txt, output_h5):
        """
        讀取 file_list_txt（裡面列出一堆 .mat 檔路徑），逐一處理，最後把所有合格 segment 寫入 output_h5
        """
        file_list = []
        with open(file_list_txt, "r", encoding="utf-8") as f_txt:
            for line in f_txt:
                line=line.strip()
                if line:
                    file_list.append(Path(line))
        print(f"\n[INFO] 即將寫入 {output_h5}, 檔案數量: {len(file_list)}")
        all_data = []
        total_segments = 0
        total_qualified = 0
        total_discarded = 0

        for fp in tqdm(file_list, desc="Processing MAT"):
            data_list, n_segments, qualified, discarded = self.process_file_1250(fp)
            total_segments += n_segments
            total_qualified += qualified
            total_discarded += discarded
            if data_list:
                all_data.extend(data_list)

        print(f"=> {output_h5} 数据条数: {len(all_data)}")
        print(f"=> 总段数: {total_segments}, 合格: {total_qualified}, 丢弃: {total_discarded}")

        if not all_data:
            print(f"=> 警告: {output_h5} 无有效数据，略过！")
            return

        # 一次性寫入 HDF5
        self.output_dir.mkdir(exist_ok=True, parents=True)
        output_path = self.output_dir / output_h5
        with h5py.File(output_path, 'w') as f_out:
            n_samples = len(all_data)
            # 建立固定大小 dataset
            f_out.create_dataset('ppg',      (n_samples,1250),     dtype='float32')
            f_out.create_dataset('ecg',      (n_samples,1250),     dtype='float32')
            f_out.create_dataset('abp',      (n_samples,1250),     dtype='float32')
            f_out.create_dataset('annotations',(n_samples,1250,4), dtype='float32')
            f_out.create_dataset('segsbp',   (n_samples,),         dtype='float32')
            f_out.create_dataset('segdbp',   (n_samples,),         dtype='float32')
            f_out.create_dataset('personal_info',(n_samples,4),    dtype='float32')
            f_out.create_dataset('vascular_properties',(n_samples,6), dtype='float32')

            for i, item in enumerate(all_data):
                f_out['ppg'][i]      = item['ppg']
                f_out['ecg'][i]      = item['ecg']
                f_out['abp'][i]      = item['abp']
                f_out['annotations'][i] = item['annotations']
                f_out['segsbp'][i]   = item['segsbp']
                f_out['segdbp'][i]   = item['segdbp']
                f_out['personal_info'][i] = item['personal_info']
                f_out['vascular_properties'][i] = item['vascular_properties']

        print(f"[INFO] {output_h5} 寫入完成。總段數={total_segments}, 合格={total_qualified}, 丟棄={total_discarded}.")


################################################
# 範例使用
################################################
if __name__=="__main__":
    # 1) 建立物件
    preparator = VitalDBDatasetPreparator(
        data_dir="PulseDB_Vital",          # 您的 .mat 檔目錄
        output_dir="training_data_VitalDB_quality2", 
        n_folds=10
    )

    # 2) 第一步：產生 folds 檔案列表 (只需執行一次)
    # preparator.create_fold_lists()

    # 3) 第二步：依某個 fold 列表產生 HDF5
    #    例如要處理 training_1_files.txt -> training_1.h5
    # 同理:
    preparator.write_h5_from_file_list("training_data_VitalDB_quality2/val_files.txt", "validation.h5")
    preparator.write_h5_from_file_list("training_data_VitalDB_quality2/test_files.txt", "test.h5")

    for i in range(1,10):
        preparator.write_h5_from_file_list(
            file_list_txt=f"training_data_VitalDB_quality2/training_{i}_files.txt",
            output_h5=f"training_{i}.h5"
        )
    

    pass
