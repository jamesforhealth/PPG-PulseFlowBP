import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_full_vascular_properties(annotations, ppg, abp=None, fs=125):
    """
    计算完整的血管属性：ptt_mean, ptt_std, pat_mean, pat_std, rr_mean, rr_std
    
    参数:
    - annotations: 形状 (N, 4) 的数组，其中包含ECG R-peaks等标记
    - ppg: PPG信号数据
    - abp: 血压信号数据(如果有)
    - fs: 采样率
    
    返回:
    - 包含6个血管属性的数组
    """
    # 获取R-peaks位置
    r_peaks = np.where(annotations[:, 0] == 1)[0]
    
    # 1-2. 计算RR intervals (mean和std)
    if len(r_peaks) < 2:
        rr_mean, rr_std = 0.0, 0.0
    else:
        rr_intervals = np.diff(r_peaks)
        rr_intervals_ms = (rr_intervals / fs) * 1000.0
        rr_mean = np.mean(rr_intervals_ms)
        rr_std = np.std(rr_intervals_ms)
    
    # 3-4. 计算PTT (Pulse Transit Time, 如果有ABP数据)
    if abp is not None and len(abp) > 0:
        # 寻找ABP信号峰值（简化版本，实际应使用更精确的方法）
        abp_peaks = []
        for i in range(1, len(abp)-1):
            if abp[i] > abp[i-1] and abp[i] > abp[i+1]:
                abp_peaks.append(i)
        
        # 计算PTT: 从R-peak到ABP峰值的时间差
        ptts = []
        for r_peak in r_peaks:
            # 找到最接近的ABP峰值
            abp_peak_after = [p for p in abp_peaks if p > r_peak]
            if abp_peak_after:
                ptt = abp_peak_after[0] - r_peak
                ptts.append((ptt / fs) * 1000.0)  # 转换为毫秒
        
        if ptts:
            ptt_mean = np.mean(ptts)
            ptt_std = np.std(ptts)
        else:
            ptt_mean, ptt_std = 0.0, 0.0
    else:
        # 如果没有ABP数据，使用默认值或已存在的值
        ptt_mean, ptt_std = 0.0, 0.0
    
    # 5-6. 计算PAT (Pulse Arrival Time, PPG峰值)
    # 寻找PPG信号峰值
    ppg_peaks = []
    for i in range(1, len(ppg)-1):
        if ppg[i] > ppg[i-1] and ppg[i] > ppg[i+1]:
            ppg_peaks.append(i)
    
    # 计算PAT: 从R-peak到PPG峰值的时间差
    pats = []
    for r_peak in r_peaks:
        # 找到最接近的PPG峰值
        ppg_peak_after = [p for p in ppg_peaks if p > r_peak]
        if ppg_peak_after:
            pat = ppg_peak_after[0] - r_peak
            pats.append((pat / fs) * 1000.0)  # 转换为毫秒
    
    if pats:
        pat_mean = np.mean(pats)
        pat_std = np.std(pats)
    else:
        pat_mean, pat_std = 0.0, 0.0
    
    # 返回所有6个属性
    return np.array([ptt_mean, ptt_std, pat_mean, pat_std, rr_mean, rr_std], dtype=np.float32)

def update_h5_with_full_vascular_properties(h5_path):
    """
    更新h5文件，添加或扩展vascular_properties到包含全部6个属性
    """
    temp_path = h5_path.parent / f"temp_{h5_path.name}"
    
    with h5py.File(h5_path, 'r') as f_in:
        n_samples = len(f_in['annotations'])
        vascular_properties = np.zeros((n_samples, 6), dtype=np.float32)
        
        # 复制现有属性(如果有)
        if 'vascular_properties' in f_in:
            old_vp = f_in['vascular_properties'][:]
            old_vp_cols = old_vp.shape[1]
            vascular_properties[:, :min(old_vp_cols, 6)] = old_vp[:, :min(old_vp_cols, 6)]
        
        # 计算缺失的属性
        for i in range(n_samples):
            annotations = f_in['annotations'][i]
            ppg = f_in['ppg'][i] if 'ppg' in f_in else None
            abp = f_in['abp'][i] if 'abp' in f_in else None
            
            # 如果已有属性，仅计算缺失部分；否则计算全部
            if 'vascular_properties' in f_in and old_vp_cols >= 6:
                continue  # 已有完整属性，跳过
            
            full_properties = compute_full_vascular_properties(annotations, ppg, abp)
            
            # 更新缺失的属性
            if 'vascular_properties' in f_in:
                for j in range(old_vp_cols, 6):
                    vascular_properties[i, j] = full_properties[j]
            else:
                vascular_properties[i] = full_properties
        
        # 创建新文件并复制所有数据
        with h5py.File(temp_path, 'w') as f_out:
            # 复制所有原始数据集(除vascular_properties外)
            for key in f_in.keys():
                if key != 'vascular_properties':
                    f_in.copy(key, f_out)
            
            # 创建新的vascular_properties
            f_out.create_dataset('vascular_properties', data=vascular_properties)
    
    # 替换原始文件
    h5_path.unlink()  # 删除原始文件
    temp_path.rename(h5_path)  # 重命名临时文件

def main():
    # 设置目录
    data_dir = Path("training_data_VitalDB_quality")
    
    # 收集所有需要处理的h5文件
    h5_files = []
    h5_files.extend(data_dir.glob("training_*.h5"))
    h5_files.extend(data_dir.glob("validation.h5"))
    h5_files.extend(data_dir.glob("test.h5"))
    
    print(f"找到 {len(h5_files)} 个h5文件需要更新")
    
    # 处理每个文件
    for h5_path in tqdm(h5_files, desc="处理h5文件"):
        try:
            update_h5_with_full_vascular_properties(h5_path)
            print(f"成功更新 {h5_path.name}")
        except Exception as e:
            print(f"处理 {h5_path.name} 时出错: {e}")

if __name__ == "__main__":
    main() 