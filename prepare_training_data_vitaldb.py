import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import scipy.signal
from collections import Counter

def create_annotation_matrix(length, peaks_dict):
    """
    建立注釋矩陣 shape=(length,4), columns=['ECG_RealPeaks','PPG_SPeaks','PPG_Turns','Normal_Signal']
    """
    # debug印一下
    anno_matrix = np.zeros((length, 4), dtype=np.float32)
    keys = ['ECG_RealPeaks', 'PPG_SPeaks', 'PPG_Turns']
    for idx, key in enumerate(keys):
        raw_peaks = peaks_dict.get(key, [])
        # 確保 raw_peaks 是整數型態
        peaks = np.array(raw_peaks, dtype=np.int64)
        # 篩選 valid
        valid_peaks = peaks[(peaks >= 0) & (peaks < length)]
        # 最後再保險一次 cast 成 int
        valid_peaks = valid_peaks.astype(np.int64)
        anno_matrix[valid_peaks, idx] = 1.0

    return anno_matrix

def find_real_peaks(sig, peaks, fs=125):
    """
    在給定 peaks 附近搜尋真正的峰值位置
    e.g. ECG_RPeaks => ECG_RealPeaks
    """
    real_peaks = []
    search_radius = int(0.1 * fs)  # 0.1 秒
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

def compute_average_time_diff(larger_array, smaller_array, sample_to_ms=8.0):
    """
    計算 average( larger - preceding_smaller ), 轉成 ms (每 sample=8ms)。
    e.g. PTT: larger=ABP_SPeaks, smaller=ECG_RealPeaks
    """
    if len(larger_array) == 0 or len(smaller_array) == 0:
        return 0.0
    la = np.sort(larger_array)
    sa = np.sort(smaller_array)
    idx_s = 0
    diffs = []
    for x in la:
        # 找 x 之前的 smaller
        while idx_s < len(sa) and sa[idx_s] < x:
            idx_s += 1
        candidate = idx_s - 1
        if candidate >= 0:
            diff = x - sa[candidate]
            diffs.append(diff)
    # 若只是 debug 印
    if len(diffs) > 0:
        return float(np.mean(diffs) * sample_to_ms)
    else:
        return 0.0

def get_scalar_from_ref(f, ref):
    """嘗試用 ref 讀取 dataset 內容，若失敗回 0.0"""
    try:
        data = f[ref][()]  # shape=? 
        return float(data[0][0])  # or data.flat[0] ...
    except:
        return 0.0

############################################
# ====== 新增 check_signal_quality() =======
def check_signal_quality(peaks_dict):
    """
    检查信号质量：
    1. 三种特征点是否按照固定顺序严格轮流出现，允许边界处存在不完整的序列。
    2. 每种特征点的间距最大值不超过最小值的两倍。
    """
    keys = ['ECG_RealPeaks', 'PPG_SPeaks', 'PPG_Turns']
    sequence_length = len(keys)

    # 确保每种特征点非空
    for key in keys:
        if len(peaks_dict[key]) == 0:
            return False

    # 检查每种特征点的间距
    for key in keys:
        intervals = np.diff(peaks_dict[key])
        if len(intervals) == 0:
            continue  # 如果只有一个峰值，无法计算间隔，但可以继续
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        if min_interval == 0:
            return False  # 间隔为零，不合理
        if max_interval > 2 * min_interval:
            return False

    # 构建按时间排序的标签序列
    all_peaks = []
    for idx, key in enumerate(keys):
        peaks = np.sort(peaks_dict[key])  # 确保峰值有序
        labels = np.full(len(peaks), idx)
        all_peaks.extend(zip(peaks, labels))
    all_peaks.sort()  # 按时间排序

    # 提取标签序列
    labels_sequence = [item[1] for item in all_peaks]

    # 需要至少有一个完整的序列
    if len(labels_sequence) < sequence_length:
        return False

    # 确定固定顺序模式
    # 取序列的第一个完整模式作为固定模式
    fixed_pattern = tuple(labels_sequence[:sequence_length])

    # 检查序列是否严格按照固定模式重复出现
    idx = 0
    while idx <= len(labels_sequence) - sequence_length:
        window = labels_sequence[idx:idx+sequence_length]
        if tuple(window) != fixed_pattern:
            return False  # 一旦发现不匹配，立即返回 False
        idx += sequence_length

    # 允许末尾存在不完整的模式
    if idx < len(labels_sequence):
        remaining = labels_sequence[idx:]
        if remaining != list(fixed_pattern)[:len(remaining)]:
            return False  # 剩余部分不匹配固定模式的前缀

    return True
################################################
# 主類別
################################################
class VitalDBDatasetPreparator:
    def __init__(self, data_dir="PulseDB_Vital", output_dir="training_data_VitalDB", n_folds=10):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_folds = n_folds

        self.total_segments   = 0
        self.total_qualified  = 0
        self.total_discarded  = 0

    def extract_peaks(self, file_handle, ref):
        """讀取 matdata 內存的對應 peaks array (可能是 object ref)"""
        data = file_handle[ref][()]
        if isinstance(data, np.ndarray):
            if data.dtype == 'object':
                # 進一步解引用
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
            # 未知類型
            peaks = np.array([], dtype=int)
        return peaks

    def process_file_1250(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                matdata = f['Subj_Wins']

                # 1) 讀 ppg, ecg, abp
                ppg_raw = []
                ecg_raw = []
                abp_raw = []
                for ref in matdata['PPG_Raw'][0]:
                    ppg_raw.append(f[ref][()].flatten())
                for ref in matdata['ECG_Raw'][0]:
                    ecg_raw.append(f[ref][()].flatten())
                for ref in matdata['ABP_Raw'][0]:
                    abp_raw.append(f[ref][()].flatten())

                ppg_raw = np.array(ppg_raw)
                ecg_raw = np.array(ecg_raw)
                abp_raw = np.array(abp_raw)

                # 2) ECG_RPeaks => find_real_peaks => ECG_RealPeaks
                ecg_realpeaks_all = []
                for idx, ref in enumerate(matdata['ECG_RPeaks'][0]):
                    approx_peaks = self.extract_peaks(f, ref)
                    real_peaks   = find_real_peaks(ecg_raw[idx], approx_peaks, fs=125)
                    ecg_realpeaks_all.append(real_peaks)
                ecg_realpeaks_all = np.array(ecg_realpeaks_all, dtype=object)

                # 3) 讀 ABP_SPeaks, ABP_Turns => compute PTT,PAT
                abp_speaks_all = []
                abp_turns_all  = []
                # ABP_SPeaks
                for ref in matdata['ABP_SPeaks'][0]:
                    peaks_abp_spk= self.extract_peaks(f, ref)
                    abp_speaks_all.append(peaks_abp_spk)
                abp_speaks_all= np.array(abp_speaks_all, dtype=object)
                # ABP_Turns
                for ref in matdata['ABP_Turns'][0]:
                    peaks_abp_tn= self.extract_peaks(f, ref)
                    abp_turns_all.append(peaks_abp_tn)
                abp_turns_all= np.array(abp_turns_all, dtype=object)

                # 4) PPG_SPeaks, PPG_Turns
                ppg_speaks_all = []
                ppg_turns_all = []
                for ref in matdata['PPG_SPeaks'][0]:
                    peaks_ppg_spk = self.extract_peaks(f, ref)
                    ppg_speaks_all.append(peaks_ppg_spk)
                ppg_speaks_all = np.array(ppg_speaks_all, dtype=object)
                for ref in matdata['PPG_Turns'][0]:
                    peaks_ppg_tn = self.extract_peaks(f, ref)
                    ppg_turns_all.append(peaks_ppg_tn)
                ppg_turns_all = np.array(ppg_turns_all, dtype=object)

                # 5) SBP, DBP
                seg_sbp= []
                seg_dbp= []
                for abp_seg in abp_raw:
                    seg_sbp.append(np.max(abp_seg))
                    seg_dbp.append(np.min(abp_seg))
                seg_sbp= np.array(seg_sbp)
                seg_dbp= np.array(seg_dbp)

                # 6) 個人 info
                age_ref    = matdata['Age'][0][0]
                gender_ref = matdata['Gender'][0][0]
                weight_ref = matdata['Weight'][0][0]
                height_ref = matdata['Height'][0][0]
                age    = get_scalar_from_ref(f, age_ref)
                gender = get_scalar_from_ref(f, gender_ref)
                weight = get_scalar_from_ref(f, weight_ref)
                height = get_scalar_from_ref(f, height_ref)
                # print(f"[Debug] age={age}, gender={gender}, weight={weight}, height={height}")

                personal_info = np.array([age, gender, weight, height], dtype=np.float32)

                # 7) 逐 segment => annotation matrix + check_signal_quality
                processed_data= []
                n_segments= len(ppg_raw)
                qualified=0
                discarded=0
                for i in range(n_segments):
                    # 建 peaks_dict
                    peaks_dict= {
                        'ECG_RealPeaks': ecg_realpeaks_all[i],
                        'PPG_SPeaks': ppg_speaks_all[i],
                        'PPG_Turns': ppg_turns_all[i]
                    }
                    # 先跑 check_signal_quality
                    if not check_signal_quality(peaks_dict):
                        discarded+=1
                        continue

                    # --- 若通過 ---
                    ptt_val= compute_average_time_diff(abp_speaks_all[i], abp_turns_all[i], sample_to_ms=8.0)
                    pat_val= compute_average_time_diff(abp_speaks_all[i], ecg_realpeaks_all[i], sample_to_ms=8.0)

                    annotation_mat= create_annotation_matrix(1250, peaks_dict)
                    data_item= {
                        'ppg': ppg_raw[i],
                        'ecg': ecg_raw[i],
                        'abp': abp_raw[i],
                        'annotations': annotation_mat,
                        'segsbp': seg_sbp[i],
                        'segdbp': seg_dbp[i],
                        'personal_info': personal_info,
                        'vascular_properties': np.array([ptt_val, pat_val], dtype=np.float32)
                    }
                    processed_data.append(data_item)
                    qualified+=1

                # 打印当前文件的统计信息
                print(f"[{file_path.name}] total={n_segments}, ok={qualified}, discard={discarded}")
                
                # **增加返回统计信息**
                return processed_data, n_segments, qualified, discarded

        except Exception as e:
            print(f"[Error] processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, 0, 0  # 返回空列表和零统计值

    def write_split_h5(self, output_path, file_list):
        print(f"\n[INFO] 正在写入 {output_path}, 文件数量: {len(file_list)}")
        all_data = []
        total_segments = 0
        total_qualified = 0
        total_discarded = 0

        for fp in tqdm(file_list):
            data_list, n_segments, qualified, discarded = self.process_file_1250(fp)
            total_segments += n_segments
            total_qualified += qualified
            total_discarded += discarded

            if data_list:
                all_data.extend(data_list)

        print(f"=> {output_path} 数据条数: {len(all_data)}")
        print(f"=> 总段数: {total_segments}, 合格: {total_qualified}, 丢弃: {total_discarded}")

        if not all_data:
            print(f"=> 警告: {output_path} 无有效数据，略过！")
            return

        with h5py.File(output_path, 'w') as f_out:
            n_samples = len(all_data)
            f_out.create_dataset('ppg',      (n_samples,1250),     dtype='float32')
            f_out.create_dataset('ecg',      (n_samples,1250),     dtype='float32')
            f_out.create_dataset('abp',      (n_samples,1250),     dtype='float32')
            f_out.create_dataset('annotations',(n_samples,1250,4), dtype='float32')
            f_out.create_dataset('segsbp',   (n_samples,),         dtype='float32')
            f_out.create_dataset('segdbp',   (n_samples,),         dtype='float32')
            f_out.create_dataset('personal_info',(n_samples,4),    dtype='float32')
            f_out.create_dataset('vascular_properties',(n_samples,2), dtype='float32')

            for i, item in enumerate(all_data):
                f_out['ppg'][i]      = item['ppg']
                f_out['ecg'][i]      = item['ecg']
                f_out['abp'][i]      = item['abp']
                f_out['annotations'][i] = item['annotations']
                f_out['segsbp'][i]   = item['segsbp']
                f_out['segdbp'][i]   = item['segdbp']
                f_out['personal_info'][i] = item['personal_info']
                f_out['vascular_properties'][i] = item['vascular_properties']

    def create_folds(self):
        files = list(self.data_dir.glob("*.mat"))
        total_files= len(files)
        np.random.shuffle(files)

        print(f"[INFO] 总共有 {total_files} 个 .mat 档.")
        fold_size= total_files // self.n_folds
        folds=[]
        for i in range(self.n_folds):
            start= i*fold_size
            end  = start+ fold_size
            if i== self.n_folds-1:
                end= total_files
            fold_files= files[start:end]
            folds.append(fold_files)
            print(f"fold{i}: {len(fold_files)} files")

        fold_idx_for_val_test= 0
        val_test_files= folds[fold_idx_for_val_test]
        half= len(val_test_files)// 2
        val_files= val_test_files[:half]
        test_files= val_test_files[half:]
        print(f"[Info] fold {fold_idx_for_val_test} => val={len(val_files)}, test={len(test_files)}")

        train_fold_count=1
        for i in range(self.n_folds):
            if i== fold_idx_for_val_test:
                continue
            out_path= self.output_dir/f"training_{train_fold_count}.h5"
            self.write_split_h5(out_path, folds[i])
            train_fold_count+=1

        val_path= self.output_dir/"validation.h5"
        test_path= self.output_dir/"test.h5"
        self.write_split_h5(val_path, val_files)
        self.write_split_h5(test_path,test_files)

        with open(self.output_dir/"val_files.txt","w",encoding="utf-8") as f_list:
            for x in val_files:
                f_list.write(f"{x.name}\n")
        with open(self.output_dir/"test_files.txt","w",encoding="utf-8") as f_list:
            for x in test_files:
                f_list.write(f"{x.name}\n")

        train_fold_count=1
        for i in range(self.n_folds):
            if i== fold_idx_for_val_test:
                continue
            with open(self.output_dir/f"training_{train_fold_count}_files.txt","w",encoding="utf-8") as f_list:
                for x in folds[i]:
                    f_list.write(f"{x.name}\n")
            train_fold_count+=1

        print("\n[INFO] Done. 产生多个 training_x.h5, 1 validation.h5, 1 test.h5.")
        print(f"总 seg: {self.total_segments}, 合格: {self.total_qualified}, 丢弃: {self.total_discarded}")

if __name__=="__main__":
    preparator= VitalDBDatasetPreparator(
        data_dir="PulseDB_Vital",  # <== 您的 .mat 档目录
        output_dir="training_data_VitalDB", 
        n_folds=10
    )
    preparator.create_folds()
