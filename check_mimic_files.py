import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_mat_file(file_path):
    """
    檢查 .mat 檔案的內容和結構
    
    參數:
    file_path: .mat 檔案的路徑
    
    返回:
    dict: 包含檢查結果的字典
    """
    try:
        with h5py.File(file_path, 'r') as f:
            results = {
                'file_name': os.path.basename(file_path),
                'status': 'OK',
                'errors': [],
                'info': {}
            }
            
            # 檢查必要的群組是否存在
            if 'Subj_Wins' not in f:
                results['errors'].append("缺少 Subj_Wins 群組")
                results['status'] = 'ERROR'
                return results
            
            matdata = f['Subj_Wins']
            
            # 檢查必要的欄位
            required_fields = [
                'PPG_Raw', 'PPG_F', 'ECG_Raw', 'ECG_F', 'ABP_Raw', 'ABP_F',
                'ECG_RPeaks', 'ECG_RealPeaks', 'PPG_SPeaks', 'PPG_Turns'
            ]
            
            for field in required_fields:
                if field not in matdata:
                    results['errors'].append(f"缺少 {field} 欄位")
                    results['status'] = 'ERROR'
                else:
                    # 檢查資料的基本結構
                    try:
                        refs = matdata[field][0]
                        n_segments = len(refs)
                        
                        # 檢查第一個片段的形狀
                        first_segment = f[refs[0]][:].shape
                        
                        results['info'][field] = {
                            'n_segments': n_segments,
                            'first_segment_shape': first_segment
                        }
                        
                        # 特別檢查 ECG_RealPeaks
                        if field == 'ECG_RealPeaks':
                            # 檢查是否有資料
                            peaks = f[refs[0]][:].flatten()
                            if len(peaks) == 0:
                                results['errors'].append(f"{field} 沒有資料")
                                results['status'] = 'WARNING'
                            
                    except Exception as e:
                        results['errors'].append(f"讀取 {field} 時發生錯誤: {str(e)}")
                        results['status'] = 'ERROR'
            
            return results
            
    except Exception as e:
        return {
            'file_name': os.path.basename(file_path),
            'status': 'ERROR',
            'errors': [f"檔案讀取錯誤: {str(e)}"],
            'info': {}
        }

def main():
    # 設定MIMIC資料集的路徑
    mimic_folder = "D:\\PulseDB\\PulseDB_MIMIC"
    
    # 確認資料夾存在
    if not os.path.exists(mimic_folder):
        print(f"錯誤: 找不到資料夾 {mimic_folder}")
        return
    
    # 獲取所有.mat檔案
    mat_files = [f for f in os.listdir(mimic_folder) if f.endswith('.mat')]
    
    # 檢查結果統計
    stats = {
        'total': len(mat_files),
        'ok': 0,
        'warning': 0,
        'error': 0,
        'files_with_errors': []
    }
    
    print(f"開始檢查 {len(mat_files)} 個檔案...")
    
    # 檢查每個檔案
    for filename in tqdm(mat_files):
        file_path = os.path.join(mimic_folder, filename)
        result = check_mat_file(file_path)
        
        if result['status'] == 'OK':
            stats['ok'] += 1
        elif result['status'] == 'WARNING':
            stats['warning'] += 1
            stats['files_with_errors'].append(result)
        else:
            stats['error'] += 1
            stats['files_with_errors'].append(result)
    
    # 輸出統計結果
    print("\n檢查結果統計:")
    print(f"總檔案數: {stats['total']}")
    print(f"正常檔案: {stats['ok']}")
    print(f"警告檔案: {stats['warning']}")
    print(f"錯誤檔案: {stats['error']}")
    
    # 輸出詳細錯誤資訊
    if stats['files_with_errors']:
        print("\n問題檔案詳細資訊:")
        for result in stats['files_with_errors']:
            print(f"\n檔案: {result['file_name']}")
            print(f"狀態: {result['status']}")
            print("錯誤:")
            for error in result['errors']:
                print(f"  - {error}")
            if result['info']:
                print("資訊:")
                for field, info in result['info'].items():
                    print(f"  {field}:")
                    for k, v in info.items():
                        print(f"    {k}: {v}")
    
    # 將檢查結果寫入檔案
    with open('mimic_files_check_report.txt', 'w', encoding='utf-8') as f:
        f.write("MIMIC 檔案檢查報告\n")
        f.write("===================\n\n")
        f.write(f"總檔案數: {stats['total']}\n")
        f.write(f"正常檔案: {stats['ok']}\n")
        f.write(f"警告檔案: {stats['warning']}\n")
        f.write(f"錯誤檔案: {stats['error']}\n\n")
        
        if stats['files_with_errors']:
            f.write("問題檔案詳細資訊:\n")
            f.write("=================\n\n")
            for result in stats['files_with_errors']:
                f.write(f"檔案: {result['file_name']}\n")
                f.write(f"狀態: {result['status']}\n")
                f.write("錯誤:\n")
                for error in result['errors']:
                    f.write(f"  - {error}\n")
                if result['info']:
                    f.write("資訊:\n")
                    for field, info in result['info'].items():
                        f.write(f"  {field}:\n")
                        for k, v in info.items():
                            f.write(f"    {k}: {v}\n")
                f.write("\n")

if __name__ == "__main__":
    main() 