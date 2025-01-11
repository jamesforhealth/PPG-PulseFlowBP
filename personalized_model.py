import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
class PersonalizedDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.ppg = f['ppg'][:]  # (N, 1250)
            self.vpg = f['vpg'][:]  # (N, 1250)
            self.apg = f['apg'][:]  # (N, 1250)
            if 'ecg' in f:
                self.ecg = f['ecg'][:]  # (N, 1250)
            else:
                print(f"[Warn] 'ecg' dataset not found in {file_path}, initializing with zeros.")
                self.ecg = torch.zeros_like(self.ppg)  # 使用與 PPG 相同的形狀初始化為零
            self.segsbp = f['segsbp'][:]  # (N,)
            self.personal_info = f['personal_info'][:]  # (N, 5)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return {
            'ppg': torch.FloatTensor(self.ppg[idx]),
            'vpg': torch.FloatTensor(self.vpg[idx]),
            'apg': torch.FloatTensor(self.apg[idx]),
            'ecg': torch.FloatTensor(self.ecg[idx]),
            'segsbp': torch.FloatTensor([self.segsbp[idx]]),
            'personal_info': torch.FloatTensor(self.personal_info[idx])
        }

class SmallBPModel(nn.Module):
    def __init__(self):
        super(SmallBPModel, self).__init__()
        self.fc1 = nn.Linear(1250 * 4 + 5, 128)  # 1250 points for each signal + 5 personal info features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, ppg, vpg, apg, ecg, personal_info):
        # Concatenate all inputs
        x = torch.cat((ppg, vpg, apg, ecg, personal_info), dim=1)  # (B, 1250 * 4 + 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(train_loader, val_loader, epochs=100, lr=1e-3):
    model = SmallBPModel()
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['ppg'], batch['vpg'], batch['apg'], batch['ecg'], batch['personal_info'])
            loss = criterion(outputs, batch['segsbp'])
            loss.backward()
            optimizer.step()

    # Validation use MAE loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['ppg'], batch['vpg'], batch['apg'], batch['ecg'], batch['personal_info'])
            val_loss += criterion2(outputs, batch['segsbp']).item()
    val_loss /= len(val_loader)
    return val_loss

if __name__ == '__main__':
    data_dir = Path("personalized_training_data_MIMIC")#Path("personalized_training_data")
    results = []

    # 遍歷所有訓練檔案
    average_loss = 0
    for train_file in tqdm(data_dir.glob("*_train.h5")):
        val_file = train_file.with_name(train_file.name.replace("_train.h5", "_val.h5"))
        
        if not val_file.exists():
            print(f"[Warn] {val_file} does not exist, skipping.")
            continue

        train_dataset = PersonalizedDataset(train_file)
        val_dataset = PersonalizedDataset(val_file)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        val_loss = train_model(train_loader, val_loader)
        results.append({
            'file': train_file.name,
            'val_loss': val_loss
        })
        print(f"Validation loss for {train_file.name}: {val_loss:.4f}")
        average_loss += val_loss

    # 將結果存入 CSV 檔案
    average_loss /= len(data_dir.glob("*_train.h5"))
    print(f"Average validation loss: {average_loss:.4f}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("personalized_training_results_MIMIC.csv", index=False)
    print("[Done] Validation losses saved to personalized_training_results.csv") 