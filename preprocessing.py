
import numpy as np
import os 
import torch
import json
import scipy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm
import sqlite3
import multiprocessing
def save_encoded_data(encoded_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for path, vectors in encoded_data.items():
        output_path = os.path.join(output_dir, path.replace('/', '_') + '.h5')
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=vectors)
            f.attrs['original_path'] = path
def get_json_files(data_folder):
    json_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def gaussian_smooth(input, window_size, sigma):
    if window_size == 0.0:
        return input
    half_window = window_size // 2
    output = np.zeros_like(input)
    weights = np.zeros(window_size)
    weight_sum = 0

    # Calculate Gaussian weights
    for i in range(-half_window, half_window + 1):
        weights[i + half_window] = np.exp(-0.5 * (i / sigma) ** 2)
        weight_sum += weights[i + half_window]

    # Normalize weights
    weights /= weight_sum

    # Apply Gaussian smoothing
    for i in range(len(input)):
        smoothed_value = 0
        for j in range(-half_window, half_window + 1):
            index = i + j
            if 0 <= index < len(input):
                smoothed_value += input[index] * weights[j + half_window]
        output[i] = smoothed_value

    # Copy border values from the input
    output[:window_size] = input[:window_size]
    output[-window_size:] = input[-window_size:]

    return output

def process_DB_rawdata(data):
    raw_data = [-value for packet in data['raw_data'] for value in packet['datas']]
    sample_rate = data['sample_rate']
    print(f'Sample_rate: {sample_rate}')
    scale = int(3 * sample_rate / 100)
    return  np.array(gaussian_smooth(raw_data, scale, scale/4), dtype=np.float32)

def add_noise_with_snr(data, target_snr_db):
    signal_power = torch.mean(data ** 2)
    signal_power_db = 10 * torch.log10(signal_power)

    noise_power_db = signal_power_db - target_snr_db
    noise_power = 10 ** (noise_power_db / 10)
    
    noise = torch.sqrt(noise_power) * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data

def add_gaussian_noise_torch(data, mean=0, std=0.002):
    """
    Add Gaussian noise to the input data using PyTorch.
    
    Args:
        data (torch.Tensor): Input data.
        mean (float): Mean of the Gaussian distribution (default is 0).
        std (float): Standard deviation of the Gaussian distribution (default is 0.1).
        
    Returns:
        torch.Tensor: Noisy data.
    """
    noise = torch.randn_like(data) * std + mean
    noisy_data = data + noise
    return noisy_data

class PulseDataset(Dataset):  
    def __init__(self, json_files, sample_rate=100):
        self.data = []
        self.sample_rate = sample_rate
        self.load_data(json_files)

    def load_data(self, json_files):
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if json_data['anomaly_list']:
                        continue
                    signal = json_data['smoothed_data']
                    original_sample_rate = json_data.get('sample_rate', 100)
                    x_points = json_data['x_points']
                    # print(f'x_points:{x_points}, json_file:{json_file}')
                    if original_sample_rate != self.sample_rate:
                        num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                        signal = scipy.signal.resample(signal, num_samples)
                        x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]
                    signal = self.normalize(signal)
                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40 : 
                            self.data.append(pulse)
                            # if len(pulse) <= 50 or len(pulse) >=  130 : input(f'pulse:{pulse}, j:{j}, pulse_start:{pulse_start}, pulse_end:{pulse_end}')
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse = self.data[idx]
        return torch.tensor(pulse, dtype=torch.float32)
    

class PulseDataset(Dataset):
    def __init__(self, json_files, target_len, sample_rate=100):
        self.data = []
        self.target_len = target_len
        self.sample_rate = sample_rate
        self.load_data(json_files)

    def load_data(self, json_files):
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if json_data['anomaly_list']:
                        continue
                    signal = json_data['smoothed_data']
                    original_sample_rate = json_data.get('sample_rate', 100)
                    x_points = json_data['x_points']

                    if original_sample_rate != self.sample_rate:
                        num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                        signal = scipy.signal.resample(signal, num_samples)
                        x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]
                    
                    signal = self.normalize(signal)
                    
                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40:
                            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, self.target_len))
                            self.data.append(pulse_resampled)
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse = self.data[idx]
        return torch.tensor(pulse, dtype=torch.float32)


def save_to_hdf5(mat_files, target_len, output_file, sample_rate=125):
    all_pulses = []

    for mat_file in tqdm(mat_files):
        try:
            with h5py.File(mat_file, 'r') as f:
                abp_raw_refs = f['Subj_Wins']['ABP_Raw'][0]
                abp_turns_refs = f['Subj_Wins']['ABP_Turns'][0]

                for i in range(len(abp_raw_refs)):
                    segment = f[abp_raw_refs[i]][:].flatten()
                    turns = f[abp_turns_refs[i]][:].flatten().astype(int)
                    
                    signal = normalize(segment)

                    # 拼接前後段訊號
                    if i > 0:
                        prev_segment = f[abp_raw_refs[i-1]][:].flatten()
                        signal = np.concatenate((prev_segment[-sample_rate:], signal))
                        turns = np.concatenate(([x + sample_rate for x in turns], turns))
                    
                    if i < len(abp_raw_refs) - 1:
                        next_segment = f[abp_raw_refs[i+1]][:].flatten()
                        signal = np.concatenate((signal, next_segment[:sample_rate]))
                        turns = np.concatenate((turns, [x + sample_rate * 2 for x in turns]))
                    
                    # 分解成一拍一拍的資料
                    for j in range(len(turns) - 1):
                        pulse_start = turns[j]
                        pulse_end = turns[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40:
                            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, target_len))
                            all_pulses.append(pulse_resampled)

        except Exception as e:
            print(f'Error in loading {mat_file}: {e}')

    all_pulses = np.array(all_pulses)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('pulses', data=all_pulses)

def normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

# class ABPPulseDataset(Dataset):
#     def __init__(self, mat_files, target_len, sample_rate=125):
#         self.data = []
#         self.target_len = target_len
#         self.sample_rate = sample_rate
#         self.load_data(mat_files)

#     def load_data(self, mat_files):
#         for mat_file in tqdm(mat_files):
#             try:
#                 with h5py.File(mat_file, 'r') as f:
#                     abp_raw_refs = f['Subj_Wins']['ABP_Raw'][0]
#                     abp_turns_refs = f['Subj_Wins']['ABP_Turns'][0]

#                     for i in range(len(abp_raw_refs)):
#                         segment = f[abp_raw_refs[i]][:].flatten()
#                         turns = f[abp_turns_refs[i]][:].flatten().astype(int)
                        
#                         signal = self.normalize(segment)

#                         # 拼接前後段訊號
#                         if i > 0:
#                             prev_segment = f[abp_raw_refs[i-1]][:].flatten()
#                             signal = np.concatenate((prev_segment[-self.sample_rate:], signal))
#                             turns = np.concatenate(([x + self.sample_rate for x in turns], turns))
                        
#                         if i < len(abp_raw_refs) - 1:
#                             next_segment = f[abp_raw_refs[i+1]][:].flatten()
#                             signal = np.concatenate((signal, next_segment[:self.sample_rate]))
#                             turns = np.concatenate((turns, [x + self.sample_rate * 2 for x in turns]))
                        
#                         # 分解成一拍一拍的資料
#                         for j in range(len(turns) - 1):
#                             pulse_start = turns[j]
#                             pulse_end = turns[j + 1]
#                             pulse = signal[pulse_start:pulse_end]
#                             if len(pulse) > 40:
#                                 interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
#                                 pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, self.target_len))
#                                 self.data.append(pulse_resampled)

#             except Exception as e:
#                 print(f'Error in loading {mat_file}: {e}')

#     def normalize(self, data):
#         return (data - np.mean(data)) / (np.std(data) + 1e-8)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         pulse = self.data[idx]
#         return torch.tensor(pulse, dtype=torch.float32)

# class ABPPulseDataset(Dataset):
#     def __init__(self, hdf5_file):
#         with h5py.File(hdf5_file, 'r') as f:
#             self.data = f['pulses'][:]
#             self.length = len(self.data)
#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         pulse = self.data[idx]
#         return torch.tensor(pulse, dtype=torch.float32)
    
# class ABPPulseDataset(Dataset):
#     def __init__(self, pulses, target_len=100):
#         self.pulses = pulses
#         self.target_len = target_len

#     def __len__(self):
#         return len(self.pulses)

#     def __getitem__(self, idx):
#         pulse = self.pulses[idx]
#         # 线性插值到目标长度
#         x = np.linspace(0, len(pulse) - 1, len(pulse))
#         f = scipy.interpolate.interp1d(x, pulse, kind='linear')
#         x_new = np.linspace(0, len(pulse) - 1, self.target_len)
#         pulse_resampled = f(x_new)
#         return torch.tensor(pulse_resampled, dtype=torch.float32)

class ABPPulseDataset(Dataset):
    def __init__(self, db_path, indices=None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        if indices is None:
            self.cursor.execute("SELECT COUNT(*) FROM pulses")
            self.length = self.cursor.fetchone()[0]
            self.indices = range(self.length)
        else:
            self.indices = indices
            self.length = len(indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        self.cursor.execute("SELECT pulse FROM pulses WHERE id=?", (real_idx+1,))
        pulse_blob = self.cursor.fetchone()[0]
        pulse = np.frombuffer(pulse_blob, dtype=np.float32)
        return torch.tensor(pulse, dtype=torch.float32)

    def close(self):
        self.conn.close()

def create_database(db_name='pulse_database.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pulses
                 (id INTEGER PRIMARY KEY,
                  mat_file TEXT,
                  segment_index INTEGER,
                  pulse_index INTEGER,
                  pulse BLOB,
                  timestamp REAL)''')
    conn.commit()
    return conn

def normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def process_mat_files(data_folder, db_conn, target_len=100, sample_rate=125):
    c = db_conn.cursor()
    mat_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.mat')]
    
    for mat_file in tqdm(mat_files):
        try:
            with h5py.File(mat_file, 'r') as f:
                abp_raw_refs = f['Subj_Wins']['ABP_Raw'][0]
                abp_turns_refs = f['Subj_Wins']['ABP_Turns'][0]

                for i in range(len(abp_raw_refs)):
                    segment = f[abp_raw_refs[i]][:].flatten()
                    turns = f[abp_turns_refs[i]][:].flatten().astype(int)
                    
                    signal = normalize(segment)

                    # 拼接前後段訊號
                    if i > 0:
                        prev_segment = f[abp_raw_refs[i-1]][:].flatten()
                        signal = np.concatenate((prev_segment[-sample_rate:], signal))
                        turns = np.concatenate(([x + sample_rate for x in turns], turns))
                    
                    if i < len(abp_raw_refs) - 1:
                        next_segment = f[abp_raw_refs[i+1]][:].flatten()
                        signal = np.concatenate((signal, next_segment[:sample_rate]))
                        turns = np.concatenate((turns, [x + sample_rate * 2 for x in turns]))
                    
                    # 分解成一拍一拍的資料
                    for j in range(len(turns) - 1):
                        pulse_start = turns[j]
                        pulse_end = turns[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40:
                            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, target_len))
                            
                            # 存储到数据库
                            c.execute("INSERT INTO pulses (mat_file, segment_index, pulse_index, pulse, timestamp) VALUES (?, ?, ?, ?, ?)",
                                      (mat_file, i, j, pulse_resampled.tobytes(), pulse_start / sample_rate))

        except Exception as e:
            print(f'Error in loading {mat_file}: {e}')
    
    db_conn.commit()


def main_data_preprocess():
    data_folder = 'D:\\PulseDB\\PulseDB_Vital'
    db_conn = create_database()
    process_mat_files(data_folder, db_conn)
    db_conn.close()


if __name__ == '__main__':
    main_data_preprocess()