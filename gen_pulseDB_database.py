import os
import h5py
import pymongo
from tqdm import tqdm

def connect_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["PulseDB"]
    return db

def extract_abp_segment(f, array_index):
    matdata = f['Subj_Wins']
    abp_raw_ref = matdata['ABP_Raw'][0][array_index]
    abp_segment = f[abp_raw_ref][:].flatten()
    abp_turns_ref = matdata['ABP_Turns'][0][array_index]
    abp_turns = f[abp_turns_ref][:].flatten().astype(int) - 1
    abp_speaks_ref = matdata['ABP_SPeaks'][0][array_index]
    abp_speaks = f[abp_speaks_ref][:].flatten().astype(int) - 1
    return abp_segment, abp_turns, abp_speaks

def process_and_store(db, file_path, patient_id, segments):
    with h5py.File(file_path, 'r') as f:
        for array_index in segments:
            abp_segment, abp_turns, abp_speaks = extract_abp_segment(f, array_index - 1)
            if len(abp_segment) == 1250:  # 確保是10秒的數據 (125 Hz * 10 s)
                record = {
                    "patient_id": patient_id,
                    "segment_index": array_index,
                    "abp_segment": abp_segment.tolist(),
                    "abp_turns": abp_turns.tolist(),
                    "abp_speaks": abp_speaks.tolist()
                }
                db.pulse_data.insert_one(record)

def main():
    PULSEDB_DIR = "D:\\PulseDB\\PulseDB_Vital"
    db = connect_mongodb()
    
    segments = get_data_segments()  # 從 sqlite3 獲取 segment 資訊
    
    for patient_id, array_index in tqdm(segments, desc="Processing segments"):
        file_path = os.path.join(PULSEDB_DIR, f"{patient_id}.mat")
        if os.path.exists(file_path):
            process_and_store(db, file_path, patient_id, [array_index])
    
    print("Data stored in MongoDB successfully.")

if __name__ == "__main__":
    main()