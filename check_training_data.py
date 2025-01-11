import h5py
from pathlib import Path
import numpy as np

def check_fold_data(fold_path):
    print(f"\n檢查 fold: {fold_path}")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        h5_path = fold_path / f'{split}.h5'
        if not h5_path.exists():
            print(f"{split}.h5 不存在")
            continue
            
        print(f"\n{split}.h5 內容:")
        print("-" * 30)
        
        with h5py.File(h5_path, 'r') as f:
            # 列出所有數據集
            print("數據集列表:")
            for key in f.keys():
                data = f[key][:]
                print(f"{key:15} | 形狀: {data.shape} | 類型: {data.dtype}")
                
                # 顯示數值範圍
                if len(data) > 0:
                    print(f"    數值範圍: [{np.min(data):.3f}, {np.max(data):.3f}]")
                    
                # 顯示前幾個樣本的血壓值
                if key == 'bp_values' and len(data) > 0:
                    print("\n    前5個樣本的血壓值 (SBP/DBP/MAP):")
                    for i in range(min(5, len(data))):
                        print(f"    樣本 {i}: {data[i]}")
                    
                    # 檢查異常值
                    invalid_bp = np.logical_or(data[:, 0] < 0, data[:, 1] < 0)  # SBP或DBP為負
                    invalid_count = np.sum(invalid_bp)
                    if invalid_count > 0:
                        print(f"\n    警告: 發現 {invalid_count} 個異常血壓值！")
                        print("    異常值示例:")
                        invalid_indices = np.where(invalid_bp)[0][:5]
                        for idx in invalid_indices:
                            print(f"    索引 {idx}: {data[idx]}")

def check_original_data(data_dir):
    print("\n檢查原始數據集的受試者信息:")
    print("=" * 50)
    
    subject_info = {}
    h5_files = list(Path(data_dir).glob('*.h5'))
    
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            # 檢查個人信息欄位
            try:
                subject_id = str(f['SubjectID'][:][0])  # 轉換為字符串
                age = float(f['Age'][:][0]) if 'Age' in f else None
                gender = str(f['Gender'][:][0]) if 'Gender' in f else None
                
                subject_info[subject_id] = {
                    'age': age,
                    'gender': gender,
                    'file': file_path.name
                }
            except Exception as e:
                print(f"處理文件 {file_path} 時出錯: {e}")
                print(f"SubjectID 內容: {f['SubjectID'][:]}")
    
    print(f"\n總共找到 {len(subject_info)} 個受試者")
    print("\n前5個受試者信息示例:")
    for i, (subject_id, info) in enumerate(list(subject_info.items())[:5]):
        print(f"\nSubject {subject_id}:")
        print(f"  年齡: {info['age']}")
        print(f"  性別: {info['gender']}")
        print(f"  文件: {info['file']}")

def main():
    # 檢查第一個fold
    fold_path = Path("training_data/fold_0")
    check_fold_data(fold_path)
    
    # 檢查原始數據集
    check_original_data("processed_data")
    
    # 顯示一些建議
    print("\n\n建議:")
    print("-" * 30)
    print("1. 需要修正血壓計算邏輯，避免出現負值")
    print("2. 建議在訓練數據中添加受試者信息")
    print("3. 確保每個樣本都能追溯到原始受試者")
    print("4. 考慮添加數據質量檢查步驟")

if __name__ == "__main__":
    main() 