import os
import h5py
import numpy as np
import sqlite3
import multiprocessing
from tqdm import tqdm

def load_annotations(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT filename, annotations FROM file_annotations")
    annotations = c.fetchall()
    conn.close()
    return {row[0]: list(row[1]) for row in annotations}

def process_mat_file(args):
    file_path, annotations = args
    try:
        with h5py.File(file_path, 'r') as f:
            matdata = f['Subj_Wins']
            data = {}
            for key in matdata.keys():  # Iterate over all keys in matdata
                if key in ['ABP_Raw', 'ECG_Raw', 'PPG_Raw']:
                    data[key] = [f[ref][:].flatten() for ref in matdata[key][0]]
                elif key in ['ABP_SPeaks', 'ABP_Turns', 'ECG_RPeaks', 'PPG_SPeaks', 'PPG_Turns']:
                    data[key] = [f[ref][:].flatten().astype(int) for ref in matdata[key][0]]
                else:
                    # Handle other data types if needed (adjust as per your requirements)
                    data[key] = [f[ref][:] for ref in matdata[key][0]]

        valid_segments = annotations.get(os.path.basename(file_path), [])
        
        valid_data = {}
        for k, v in data.items():
            # print(f'k:{k}')
            valid_data[k] = []
            for i, seg in enumerate(v):
                if i < len(valid_segments) and valid_segments[i] == 0:
                    valid_data[k].append(seg)
                    # input(f'i: {i}, seg:{seg}')
        if not any(valid_data.values()):
            print(f"Skipping {file_path}: No valid segments found")
            return None

        return valid_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def save_data(data, annotations, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.h5")

    with h5py.File(output_path, 'w') as f:
        for key, segments in data.items():
            # Check for uniform segment shapes:
            if all(seg.shape == segments[0].shape for seg in segments):
                f.create_dataset(key, data=segments)
            else:
                # Handle ragged arrays (segments with different shapes)
                dt = h5py.special_dtype(vlen=segments[0].dtype)
                dset = f.create_dataset(key, (len(segments),), dtype=dt)
                for i, seg in enumerate(segments):
                    dset[i] = seg

        f.create_dataset("annotations", data=np.array(annotations))
# def save_data(data, annotations, output_dir, filename):
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{filename}.h5")
#     with h5py.File(output_path, 'w') as f:
#         for key, segments in data.items():
#             f.create_dataset(key, data=segments)
#         f.create_dataset("annotations", data=np.array(annotations))

# def process_database(data_source, db_name, output_dir):
#     annotations = load_annotations(db_name)

#     for filename in tqdm(os.listdir(data_source)):
#         if filename.endswith('.mat'):
#             file_path = os.path.join(data_source, filename)

#             data = process_mat_file((file_path, annotations))
#             print(f'Processing {file_path}, data: {data}')
#             if data:
#                 valid_indices = [i for i, status in enumerate(annotations[filename]) if status == 0]
#                 save_data(data, valid_indices, output_dir, filename)

def process_database(data_source, db_name, output_dir):
    annotations = load_annotations(db_name)

    for filename in os.listdir(data_source):
        if filename.endswith('.mat'):
            file_path = os.path.join(data_source, filename)

            if filename not in annotations:
                print(f"Warning: No annotations found for {filename}. Skipping.")
                continue  # Skip to the next file

            data = process_mat_file((file_path, annotations))
            print(f'Processing {file_path}, data: {data}')
            if data:
                valid_indices = [i for i, status in enumerate(annotations[filename]) if status == 0]
                save_data(data, valid_indices, output_dir, filename)

if __name__ == "__main__":
    data_source = "PulseDB_Vital"
    db_name = "pulsedb_annotations_vital.db"
    output_dir = "processed_data"
    process_database(data_source, db_name, output_dir)
