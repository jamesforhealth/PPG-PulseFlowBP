import h5py
from pathlib import Path
import numpy as np

def check_h5_file(file_path):
    print(f"\n檢查文件: {file_path}")
    print("-" * 50)
    
    with h5py.File(file_path, 'r') as f:
        # 列出所有數據集
        print("數據集列表:")
        for key in f.keys():
            data = f[key][:]
            if isinstance(data, np.ndarray):
                shape_str = str(data.shape)
            else:
                shape_str = "標量值"
            
            print(f"{key:15} | 形狀: {shape_str:30} | 類型: {data.dtype}")
            
            # 顯示個人資訊的具體值
            if key in ['Age', 'BMI', 'CaseID', 'Gender', 'Height', 'SubjectID', 
                      'Weight', 'SegDBP', 'SegSBP']:
                if isinstance(data, np.ndarray):
                    value = data.flatten()[0] if data.size > 0 else "空"
                else:
                    value = data
                print(f"    值: {value}")
        
        # 檢查必要欄位是否存在
        required_fields = ['Age', 'BMI', 'Gender', 'Height', 'Weight', 
                         'SegDBP', 'SegSBP', 'SubjectID',
                         'ABP_Raw', 'ECG_Raw', 'PPG_Raw']
        missing_fields = [field for field in required_fields if field not in f]
        
        if missing_fields:
            print("\n缺失的必要欄位:")
            for field in missing_fields:
                print(f"- {field}")
        
        # 檢查信號數據的基本統計
        for signal in ['ABP_Raw', 'ECG_Raw', 'PPG_Raw']:
            if signal in f:
                data = f[signal][:]
                print(f"\n{signal} 統計信息:")
                print(f"片段數量: {len(data)}")
                if len(data) > 0:
                    if isinstance(data[0], np.ndarray):
                        print(f"每個片段長度: {len(data[0])}")
                    print(f"數值範圍: [{np.min(data):.3f}, {np.max(data):.3f}]")

def main():
    data_dir = Path("processed_data")
    h5_files = list(data_dir.glob("*.h5"))
    
    print(f"找到 {len(h5_files)} 個 h5 文件")
    
    # 檢查前5個文件
    for file_path in h5_files[:5]:
        check_h5_file(file_path)
        
    # 統計所有文件的欄位情況
    print("\n\n所有文件的欄位統計:")
    print("-" * 50)
    field_stats = {}
    
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if key not in field_stats:
                    field_stats[key] = 0
                field_stats[key] += 1
    
    for field, count in field_stats.items():
        print(f"{field:15} | 出現在 {count}/{len(h5_files)} 個文件中")

if __name__ == "__main__":
    main()
