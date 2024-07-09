import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import os
import torch.nn.functional as F
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
        
    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        x = self.encoder(x)
        # print(f'encoder.shape: {x.shape}')
        x = self.decoder(x)
        # print(f'decoder.shape: {x.shape}')
        return x

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels)
#             )

#     def forward(self, x):
#         # print(f'x.shape: {x.shape}')
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         # print(f'out.shape: {out.shape}')
#         return out

# class ConvAutoencoder2(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder2, self).__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),  # 1251 -> 626
#             ResidualBlock(16, 16),
#             nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 627 -> 314
#             ResidualBlock(32, 32),
#             nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 315 -> 158
#             ResidualBlock(64, 64),
#             nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # 159 -> 80
#             ResidualBlock(128, 128)
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             ResidualBlock(128, 128),
#             nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 80 -> 159
#             ResidualBlock(64, 64),
#             nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 159 -> 317
#             ResidualBlock(32, 32),
#             nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 317 -> 633
#             ResidualBlock(16, 16),
#             nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),  # 633 -> 1265
#             # nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         # print(f'encoder.shape: {x.shape}')
#         x = self.decoder(x)
#         # print(f'decoder.shape: {x.shape}')
#         # 拉伸到1250
#         x = F.interpolate(x, size=1250, mode='linear', align_corners=False)
#         return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.001):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ConvAutoencoder2(nn.Module):
    def __init__(self, dropout_rate=0.001):
        super(ConvAutoencoder2, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(4, 4, dropout_rate=dropout_rate),
            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(8, 8, dropout_rate=dropout_rate),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(16, 16, dropout_rate=dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(32, 32, dropout_rate=dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(64, 64, dropout_rate=dropout_rate)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(64, 64, dropout_rate=dropout_rate),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(32, 32, dropout_rate=dropout_rate),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(16, 16, dropout_rate=dropout_rate),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(8, 8, dropout_rate=dropout_rate),
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.Dropout(dropout_rate),
            ResidualBlock(4, 4, dropout_rate=dropout_rate),
            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=1250, mode='linear', align_corners=False)
        return x

def add_noise(batch):
    noise = torch.randn_like(batch) 
    return batch + noise


def load_data(file_path='training_set.npz'):
    data = np.load(file_path, allow_pickle=True)['abp_segments']
    print(f"Loaded data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Sample of data:\n{data[:2, :10]}")  # 打印前两行的前10个元素
    return data

# 数据预处理
def preprocess_data(data):
    data = np.array(data)
    print(f'data: {data}, shape: {data.shape}')

    # 标准化
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    return data

# 训练模型
def train_model(model, train_loader, val_loader, target_model_path, num_epochs, learning_rate=1e-4):
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    min_val_loss = float('inf')
    
    # 第一阶段：MAE训练
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = mae_criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                outputs = model(batch)
                loss = mae_criterion(outputs, batch)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'MAE Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}')
        
        if avg_val_loss < min_val_loss * 0.95:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), target_model_path)
            print(f"Saved model parameters at epoch {epoch+1}, Val Loss: {avg_val_loss:.8f}")

    # 第二阶段：MSE训练
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = mse_criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                outputs = model(batch)
                loss = mse_criterion(outputs, batch)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'MSE Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}')
        
        if avg_val_loss < min_val_loss * 0.95:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), target_model_path)
            print(f"Saved model parameters at epoch {epoch+1}, Val Loss: {avg_val_loss:.8f}")

def k_fold_cross_validation(data_tensor, model, k=20, num_epochs=10000, learning_rate=1e-4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_tensor)):
        print(f"Fold {fold+1}/{k}")
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(TensorDataset(data_tensor), batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(TensorDataset(data_tensor), batch_size=32, sampler=val_sampler)
        
        model = ConvAutoencoder2().to(device)
        target_model_path = f'abp1250model2_fold{fold+1}.pt'
            
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        train_model(model, train_loader, val_loader, target_model_path, num_epochs, learning_rate)
        
        # 评估模型
        model.load_state_dict(torch.load(target_model_path))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                outputs = model(batch)
                loss = nn.MSELoss()(outputs, batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        fold_results.append(avg_val_loss)
        print(f"Fold {fold+1} validation loss: {avg_val_loss:.8f}")
    
    return fold_results


# 异常检测
def detect_anomalies(model, data, threshold):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        mse = nn.MSELoss(reduction='none')
        reconstruction_errors = mse(reconstructed, data).mean(dim=(1,2))
    return reconstruction_errors > threshold


def predict_reconstructed_1250abp(abp_signal):
    # 确保模型和数据在同一设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = ConvAutoencoder2().to(device)
    model.load_state_dict(torch.load('abp1250model2.pt', map_location=device))
    # model.load_state_dict(torch.load('abp1250model2_fold5.pt', map_location=device))
    model.eval()

    # 预处理输入数据
    abp_signal = np.array(abp_signal)
    if abp_signal.shape[0] != 1250:
        raise ValueError("Input signal must have 1250 points")
    
    # 标准化
    # abp_signal = (abp_signal - np.mean(abp_signal)) / np.std(abp_signal)
    
    # 转换为PyTorch张量并添加批次和通道维度
    input_tensor = torch.FloatTensor(abp_signal).unsqueeze(0).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        reconstructed = model(input_tensor)
    
    # 将结果转回numpy数组并去除多余的维度
    reconstructed_abp = reconstructed.squeeze().cpu().numpy()

    # 反标准化（可选，取决于您是否需要原始信号的尺度）
    # reconstructed_abp = reconstructed_abp * np.std(abp_signal) + np.mean(abp_signal)

    #compute loss
    mse = nn.MSELoss(reduction='none')
    reconstruction_errors = mse(reconstructed, input_tensor)#.mean(dim=(1,2))


    return reconstructed_abp, reconstruction_errors.squeeze().cpu().numpy()

# 主函数
def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载和预处理数据
    data = load_data()
    print(f"Any NaN values: {np.isnan(data).any()}")
    print(f"Any infinite values: {np.isinf(data).any()}")
    # data = np.array(data)
    print(f'data: {data}, shape: {data.shape}, type: {type(data)}')
    # data = preprocess_data(data)
    
    # 转换为PyTorch张量
    data_tensor = torch.FloatTensor(data).unsqueeze(1)  # 添加通道维度
    print(f"Data tensor shape: {data_tensor.shape}")
    # fold_results = k_fold_cross_validation(data_tensor, ConvAutoencoder2(), k=10)
    # # 输出总体结果
    # print(f"Average validation loss across all folds: {np.mean(fold_results):.8f}")
    # print(f"Standard deviation of validation loss: {np.std(fold_results):.8f}")
    
    # # 选择最佳模型
    # best_fold = np.argmin(fold_results)
    # best_model_path = f'abp1250model2_fold{best_fold+1}.pt'
    # print(f"Best model is from fold {best_fold+1} with validation loss: {fold_results[best_fold]:.8f}")
    
    # # 加载最佳模型并进行最终评估
    # best_model = ConvAutoencoder2().to(device)
    # best_model.load_state_dict(torch.load(best_model_path))
    
    # # 这里可以添加对最佳模型的最终评估代码
    
    # print("Training and evaluation k-fold completed.")


    dataset_size = len(data_tensor)
    validation_split = 0.05
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(data_tensor), batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(TensorDataset(data_tensor), batch_size=32, sampler=valid_sampler)
#  # 初始化模型
    model = ConvAutoencoder2().to(device)
    target_model_path = 'abp1250model3.pt'
    
    # 计算参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    
    # 训练模型
    train_model(model, train_loader, val_loader, target_model_path, num_epochs=5000, learning_rate=1e-5)
    
    print("Training completed. Model saved as abp1250model3.pt")
    
    # 加载训练好的模型进行评估
    model.load_state_dict(torch.load(target_model_path))
    model.eval()
    
    # 在验证集上进行最终评估
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch[0].to(device)
            outputs = best_model(batch)
            loss = nn.MSELoss()(outputs, batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Final validation loss: {avg_val_loss:.8f}")
    # # 创建数据加载器
    # train_loader = DataLoader(TensorDataset(data_tensor), batch_size=32, shuffle=True)
    
    # # 初始化和训练模型
    # model = ConvAutoencoder().to(device)
    # model = ConvAutoencoder2().to(device)
    # target_model_path = 'abp1250model2.pt'
    
    # # 嘗試載入已有的模型參數
    # if os.path.exists(target_model_path):
    #     model.load_state_dict(torch.load(target_model_path))
    #     print(f"Loaded model parameters from {target_model_path}")    
    # # 計算參數量
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")

    # train_model(model, train_loader, target_model_path)
    
    # # # 保存模型
    # # torch.save(model.state_dict(), 'abp1250model.pt')
    
    # # 计算重构误差阈值（例如，使用95%分位数）
    # model.eval()
    # with torch.no_grad():
    #     reconstructed = model(data_tensor.to(device))
    #     mse = nn.MSELoss(reduction='none')
    #     reconstruction_errors = mse(reconstructed, data_tensor.to(device)).mean(dim=(1,2)).cpu().numpy()
    # threshold = np.percentile(reconstruction_errors, 98)
    
    # print(f"Reconstruction error threshold: {threshold}")
    
    # # 示例：检测异常
    # anomalies = detect_anomalies(model, data_tensor.to(device), threshold)
    # print(f"Number of detected anomalies: {anomalies.sum().item()}")


    # # 保存异常数据
    # anomaly_data = data[anomalies.cpu().numpy()]
    # np.savez('anomaly_data.npz', abp_segments=anomaly_data)
    # print(f"Saved {len(anomaly_data)} anomaly segments to anomaly_data.npz")



if __name__ == "__main__":
    main()