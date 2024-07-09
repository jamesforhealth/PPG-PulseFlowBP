


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import os
import sys
import json
import random
import scipy
from torchviz import make_dot
from preprocessing import get_json_files, PulseDataset, ABPPulseDataset, save_to_hdf5
from load_normap_ABP_data import get_data_segments, create_training_set, load_training_set
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import sqlite3


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            pulse = batch.to(device)
            output, _ = model(pulse)
            loss = criterion(output, pulse)
            total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss

# def train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=10000, save_interval=1):
#     model.train()
#     min_loss = float('inf')
#     model_path = 'init_baseline.pth'
#     target_model_path = 'ABP_autoencoder.pth'
    
#     # 嘗試載入已有的模型參數
#     if os.path.exists(target_model_path):
#         model.load_state_dict(torch.load(target_model_path))
#         print(f"Loaded model parameters from {target_model_path}")

#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in train_dataloader:
#             pulse = batch.to(device)
#             optimizer.zero_grad()
#             output, _ = model(pulse)
#             loss = criterion(output, pulse)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         total_loss /= len(train_dataloader)
        
#         test_loss = evaluate_model(model, test_dataloader, criterion, device)
#         print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss:.10f}, Testing Loss: {test_loss:.10f}")

#         # Save model parameters if test loss decreases
#         if (epoch + 1) % save_interval == 0 and test_loss < min_loss * 0.95:
#             min_loss = test_loss
#             torch.save(model.state_dict(), target_model_path)
#             print(f"Saved model parameters at epoch {epoch+1}, Testing Loss: {test_loss:.10f}")
#         # Save model parameters every save_interval epochs
#         # elif (epoch + 1) % save_interval == 0:
#         #     torch.save(model.state_dict(), model_path)

def train_autoencoder(model, train_indices, test_indices, db_path, optimizer, criterion, device, epochs=10000, batch_size=32, save_interval=1):
    model.train()
    min_loss = float('inf')
    target_model_path = 'ABP_autoencoder.pth'
    
    train_dataset = ABPPulseDataset(db_path, train_indices)
    test_dataset = ABPPulseDataset(db_path, test_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            pulse = batch.to(device)
            optimizer.zero_grad()
            output, _ = model(pulse)
            loss = criterion(output, pulse)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_dataloader)
        
        test_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss:.10f}, Testing Loss: {test_loss:.10f}")

        if (epoch + 1) % save_interval == 0 and test_loss < min_loss * 0.95:
            min_loss = test_loss
            torch.save(model.state_dict(), target_model_path)
            print(f"Saved model parameters at epoch {epoch+1}, Testing Loss: {test_loss:.10f}")

    train_dataset.close()
    test_dataset.close()

class EPGBaselinePulseAutoencoder(nn.Module):
    def __init__(self, target_len, hidden_dim=50, latent_dim=30):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(target_len, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_len)
        )

    def forward(self, x):
        z = self.enc(x)
        pred = self.dec(z)
        return pred, z

def predict_reconstructed_signal(signal, sample_rate, peaks):
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pulse_interpolate_autoencoder.pth'
    model = EPGBaselinePulseAutoencoder(target_len).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 重採樣信號
    resample_ratio = 1.0
    if sample_rate != 100:
        resample_ratio = 100 / sample_rate
        signal = scipy.signal.resample(signal, int(len(signal) * resample_ratio))
        peaks = [int(p * resample_ratio) for p in peaks]  # 調整peaks索引

    # 全局標準化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std

    # 逐拍重建
    reconstructed_signal = np.copy(signal)
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 記錄脈衝的原始長度

        if pulse_length > 1:
            # 插值到目標長度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_pulse, _ = model(pulse_tensor)
                reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()

            # 將重建的脈衝還原為原始長度
            interp_func_reconstructed = scipy.interpolate.interp1d(np.linspace(0, target_len - 1, target_len), reconstructed_pulse, kind='linear', fill_value="extrapolate")
            reconstructed_pulse_resampled = interp_func_reconstructed(np.linspace(0, target_len - 1, pulse_length))
            reconstructed_signal[start_idx:end_idx] = reconstructed_pulse_resampled

    # 反標準化
    reconstructed_signal = reconstructed_signal * std + mean

    # 根據原始採樣率調整重構信號的長度
    original_length = int(len(reconstructed_signal) / resample_ratio)
    reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    return reconstructed_signal

def predict_reconstructed_abp(signal, abp_turns, sample_rate=125, model_path='init_baseline.pth'):
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EPGBaselinePulseAutoencoder(target_len).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 全局標準化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std

    # 逐拍重建
    reconstructed_signal = np.copy(signal)
    for i in range(len(abp_turns) - 1):
        start_idx = abp_turns[i]
        end_idx = abp_turns[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 記錄脈衝的原始長度

        if pulse_length > 1:
            # 插值到目標長度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_pulse, _ = model(pulse_tensor)
                reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()

            # 將重建的脈衝還原為原始長度
            interp_func_reconstructed = scipy.interpolate.interp1d(np.linspace(0, target_len - 1, target_len), reconstructed_pulse, kind='linear', fill_value="extrapolate")
            reconstructed_pulse_resampled = interp_func_reconstructed(np.linspace(0, target_len - 1, pulse_length))
            reconstructed_signal[start_idx:end_idx] = reconstructed_pulse_resampled

    # 反標準化
    reconstructed_signal = reconstructed_signal * std + mean

    return reconstructed_signal


def main():
    data_folder = 'D:\\PulseDB\\PulseDB_Vital'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    mat_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.mat')]
    db_path = 'pulse_database.db'
    # 設置參數
    target_len = 200
    batch_size = 32
    lr = 1e-3
    # output_file = 'processed_pulses.h5'
    # save_to_hdf5(mat_files, 100, output_file)


    # 加載並劃分數據集
    # dataset = ABPPulseDataset(mat_files, target_len)
    # dataset_h_path = 'processed_pulses.h5'
    # dataset = ABPPulseDataset(dataset_h_path)
    # train_data, test_data = train_test_split(dataset, test_size=0.001, random_state=42)   
    
    # segments = get_data_segments()
    # create_training_set(segments)
    # pulses = load_training_set()
    # dataset = ABPPulseDataset(pulses)
    # train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
    
    # train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # # 印出訓練資料的長度大小
    # print(f'train_data: {len(train_data)}, test_data: {len(test_data)}')
    
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pulses")
    total_count = cursor.fetchone()[0]
    conn.close()
    print(f'total_count: {total_count}')

    # 划分训练集和测试集
    all_indices = list(range(total_count))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.001, random_state=42)

    # 初始化模型和優化器
    model = EPGBaselinePulseAutoencoder(target_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    # train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device)
    train_autoencoder(model, train_indices, test_indices, db_path, optimizer, criterion, device)

if __name__ == '__main__':
    main()