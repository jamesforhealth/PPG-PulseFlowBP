import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from prepare_training_data import calculate_first_derivative, calculate_second_derivative, encode_personal_info
from prepare_training_data_vitaldb import VitalDBDatasetPreparator
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def write_single_mat_as_h5(preparator, mat_path, out_path):
    """
    將單一 mat 檔 => 轉成單一 .h5 (包含若干 segments),
    內部欄位與原先相同: ppg, ecg, annotations, segsbp, segdbp, ...
    """
    # 檢查輸出檔案是否已存在
    out_path = Path(out_path)
    if out_path.exists():
        logger.info(f"=> {out_path.name} 已存在，跳過處理")
        return
    
    # 重用 preprocess 邏輯
    # 讀取並 process
    data_list, n_segments, qualified, discarded = preparator.process_file_1250(mat_path)
    print(f"[{mat_path.name}] => segments={n_segments}, ok={qualified}, discard={discarded}")
    if len(data_list)==0:
        print("=> 無有效 data, skip.")
        return
    
    # 寫 HDF5
    with h5py.File(out_path, 'w') as f_out:
        n_samples = len(data_list)
        f_out.create_dataset('ppg', (n_samples,1250), dtype='float32')
        f_out.create_dataset('ecg', (n_samples,1250), dtype='float32')
        f_out.create_dataset('abp', (n_samples,1250), dtype='float32')
        f_out.create_dataset('annotations',(n_samples,1250,4), dtype='float32')
        f_out.create_dataset('segsbp',(n_samples,), dtype='float32')
        f_out.create_dataset('segdbp',(n_samples,), dtype='float32')
        f_out.create_dataset('personal_info',(n_samples,4), dtype='float32')
        f_out.create_dataset('vascular_properties',(n_samples,2), dtype='float32')

        for i, item in enumerate(data_list):
            f_out['ppg'][i]      = item['ppg']
            f_out['ecg'][i]      = item['ecg']
            f_out['abp'][i]      = item['abp']
            f_out['annotations'][i] = item['annotations']
            f_out['segsbp'][i]   = item['segsbp']
            f_out['segdbp'][i]   = item['segdbp']
            f_out['personal_info'][i] = item['personal_info']
            f_out['vascular_properties'][i] = item['vascular_properties']
    print(f"=> Done writing {out_path}")

if __name__=="__main__":
    # 注意：要先初始化 preparator
    preparator = VitalDBDatasetPreparator(
        data_dir="PulseDB_Vital",          # 您的 .mat 檔目錄
        output_dir="personalized_training_data_VitalDB", 
        n_folds=10
    )
    # with open("training_data_VitalDB_quality/test_files.txt","r",encoding="utf-8") as f:
    #     for line in tqdm(f):
    #         matfile = Path(line.strip())
    #         print(f'Processing {matfile}')
    #         # 例如 out => "personalized_training_data_VitalDB/xxx.h5"
    #         outname = matfile.stem + ".h5"
    #         outpath = Path("personalized_training_data_VitalDB") / outname
    #         write_single_mat_as_h5(preparator, matfile, outpath)
    
    with open("training_data_VitalDB_quality/val_files.txt","r",encoding="utf-8") as f:
        for line in tqdm(f):
            matfile = Path(line.strip())
            print(f'Processing {matfile}')
            # 例如 out => "personalized_training_data_VitalDB/xxx.h5"
            outname = matfile.stem + ".h5"
            outpath = Path("personalized_training_data_VitalDB") / outname
            write_single_mat_as_h5(preparator, matfile, outpath)
    
    for i in range(1, 10):
        with open(f"training_data_VitalDB_quality/training_{i}_files.txt","r",encoding="utf-8") as f:
            for line in tqdm(f):
                matfile = Path(line.strip())
                print(f'Processing {matfile}')
                # 例如 out => "personalized_training_data_VitalDB/xxx.h5"
                outname = matfile.stem + ".h5"
                outpath = Path("personalized_training_data_VitalDB") / outname
                write_single_mat_as_h5(preparator, matfile, outpath)

