import h5py
import os

def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"文件: {file_path}")
        print("数据结构:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  - {name}: 形状 {obj.shape}, 类型 {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  + {name}")
        
        f.visititems(print_structure)
        input("\n")

# 遍历processed_data文件夹中的所有h5文件
processed_data_folder = "processed_data"
for filename in os.listdir(processed_data_folder):
    if filename.endswith('.h5'):
        file_path = os.path.join(processed_data_folder, filename)
        inspect_h5_file(file_path)