import sqlite3
import h5py
import numpy as np
import os

PULSEDB_DIR = "D:\\PulseDB\\PulseDB_Vital"

def get_data_segments():
    conn = sqlite3.connect('PulseDB analysis test3.sqlite3')
    cursor = conn.cursor()
    cursor.execute("""
    SELECT p.identifier, d.array_index
    FROM data_segment d
    JOIN patient_info_snapshot p ON d.patient_snapshot_id = p.id
    """)
    segments = cursor.fetchall()
    conn.close()
    return segments

def extract_abp_segment(mat_file, array_index):
    with h5py.File(mat_file, 'r') as f:
        matdata = f['Subj_Wins']
        abp_raw_ref = matdata['ABP_Raw'][0][array_index]
        abp_segment = f[abp_raw_ref][:].flatten()
        abp_turns_ref = matdata['ABP_Turns'][0][array_index]
        abp_turns = f[abp_turns_ref][:].flatten().astype(int) - 1  # 将turn点的索引减1
        abp_speaks_ref = matdata['ABP_SPeaks'][0][array_index]
        abp_speaks = f[abp_speaks_ref][:].flatten().astype(int) - 1  # 将speak点的索引减1
    return abp_segment, abp_turns, abp_speaks

def create_training_set(segments, output_file='training_set.npz'):
    abp_segments = []
    abp_turns_list = []
    abp_speaks_list = []
    for patient_id, array_index in segments:
        mat_file = os.path.join(PULSEDB_DIR, f"{patient_id}.mat")
        if os.path.exists(mat_file):
            abp_segment, abp_turns, abp_speaks = extract_abp_segment(mat_file, array_index - 1)
            if len(abp_segment) == 1250:  # 确保是10秒的数据 (125 Hz * 10 s)
                abp_segments.append(abp_segment)
                abp_turns_list.append(abp_turns)
                abp_speaks_list.append(abp_speaks)
    
    abp_segments = np.array(abp_segments, dtype=np.float32)
    abp_turns_list = np.array(abp_turns_list, dtype=object)
    abp_speaks_list = np.array(abp_speaks_list, dtype=object)
    
    np.savez(output_file, abp_segments=abp_segments, abp_turns=abp_turns_list, abp_speaks=abp_speaks_list)
    
    print(f"Saved {len(abp_segments)} input data to {output_file}")

def load_training_set(file_path='training_set.npz'):
    data = np.load(file_path, allow_pickle=True)
    return data['abp_segments'], data['abp_turns'], data['abp_speaks']

def main():
    segments = get_data_segments()
    create_training_set(segments)
    abp_segments, abp_turns, abp_speaks = load_training_set()
    print(f"Loaded {len(abp_segments)} segments, shape: {abp_segments.shape}")
    print(f"Number of turn points: {len(abp_turns)}")
    print(f"Number of speak points: {len(abp_speaks)}")

if __name__ == "__main__":
    main()