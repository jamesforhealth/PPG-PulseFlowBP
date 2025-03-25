import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from pathlib import Path
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
import os
from sklearn.model_selection import train_test_split
import random

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

################################################
# 1. 数据集定义
################################################

class ECGDataset(Dataset):
    """
    从h5文件中加载ECG数据和R-Peaks标注
    
    参数:
        h5_file: h5文件路径
    """
    def __init__(self, h5_file, transform=None):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        
        # 打开h5文件获取数据集大小，然后关闭
        with h5py.File(h5_file, 'r') as h5:
            if 'ecg' not in h5:
                raise ValueError(f"在{h5_file}中找不到ECG数据")
            if 'annotations' not in h5:
                raise ValueError(f"在{h5_file}中找不到annotations数据")
            self.length = len(h5['ecg'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 每次获取数据时打开h5文件
        with h5py.File(self.h5_file, 'r') as h5:
            # 获取ECG数据
            ecg_data = h5['ecg'][idx]  # (1250,)
            
            # 获取annotations数据
            anno_data = h5['annotations'][idx]  # (1250, 4)
            
            # 提取R-Peaks标注 (假设R-Peaks在annotations的第一个通道)
            r_peaks_mask = anno_data[:, 0]  # (1250,)
            
            # 找到R-Peaks的位置
            r_peaks = np.where(r_peaks_mask > 0)[0]
            
            # 转换为PyTorch张量
            ecg_tensor = torch.from_numpy(ecg_data).float().unsqueeze(0)  # (1, 1250)
            r_peaks_mask_tensor = torch.from_numpy(r_peaks_mask).float().unsqueeze(0)  # (1, 1250)
            
            # 应用数据增强
            if self.transform:
                ecg_tensor = self.transform(ecg_tensor)
            
            return {
                'ecg': ecg_tensor,
                'label': r_peaks_mask_tensor,
                'r_peaks': r_peaks
            }

################################################
# 2. 模型定义
################################################

class ResidualBlock(nn.Module):
    """
    TCN的残差块，包含扩张卷积和残差连接
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        # 计算填充以保持序列长度
        # 对于奇数kernel_size，padding = (kernel_size - 1) * dilation // 2
        # 对于偶数kernel_size，需要不同的左右填充
        padding = (kernel_size - 1) * dilation // 2
        
        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
        # 确保输出尺寸与输入相同
        self.same_size = True
        self.crop = None
    
    def forward(self, x):
        residual = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 如果尺寸不匹配，裁剪输出以匹配输入尺寸
        if out.size(2) != x.size(2):
            # 居中裁剪
            diff = out.size(2) - x.size(2)
            if diff > 0:
                # 输出太长，需要裁剪
                out = out[:, :, diff//2:-(diff-diff//2)]
            else:
                # 输出太短，需要填充（这种情况不应该发生）
                diff = abs(diff)
                out = F.pad(out, (diff//2, diff-diff//2))
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        
        return out

class TCN(nn.Module):
    """
    时间卷积网络，由多个残差块组成
    """
    def __init__(self, input_channels, output_channels, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation_size, dropout))
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Conv1d(num_channels[-1], output_channels, 1)
    
    def forward(self, x):
        out = self.network(x)
        out = self.linear(out)
        return out

class SelfAttention(nn.Module):
    """
    自注意力机制，用于捕获序列内的长距离依赖关系
    """
    def __init__(self, channels, reduction=8):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(channels, channels // reduction, 1)
        self.key = nn.Conv1d(channels, channels // reduction, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, length = x.size()
        
        # 计算查询、键和值
        query = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # B x L x C'
        key = self.key(x).view(batch_size, -1, length)  # B x C' x L
        value = self.value(x)  # B x C x L
        
        # 计算注意力权重
        attention = torch.bmm(query, key)  # B x L x L
        attention = F.softmax(attention, dim=2)
        
        # 应用注意力权重
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x L
        
        # 残差连接
        out = self.gamma * out + x
        
        return out, attention

class BiLSTM(nn.Module):
    """
    双向LSTM，用于捕获长距离依赖关系
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
    
    def forward(self, x):
        # x: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM forward
        out, _ = self.lstm(x)
        
        # 转回原来的形状: (batch_size, seq_len, 2*hidden_size) -> (batch_size, 2*hidden_size, seq_len)
        out = out.permute(0, 2, 1)
        
        return out

class ECGPeakDetector(nn.Module):
    """
    ECG R-Peak检测模型，结合TCN、注意力机制和BiLSTM
    """
    def __init__(self, input_channels=1, output_channels=1, hidden_size=64):
        super(ECGPeakDetector, self).__init__()
        
        # TCN部分
        num_channels = [32, 64, 128, 256]
        kernel_size = 5
        dropout = 0.2
        self.tcn = TCN(input_channels, hidden_size, num_channels, kernel_size, dropout)
        
        # 注意力机制
        self.attention = SelfAttention(hidden_size)
        
        # BiLSTM部分
        self.bilstm = BiLSTM(hidden_size, hidden_size//2)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, output_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # TCN
        tcn_out = self.tcn(x)
        
        # 注意力机制
        att_out, attention_map = self.attention(tcn_out)
        
        # BiLSTM
        lstm_out = self.bilstm(att_out)
        
        # 合并特征
        combined = torch.cat([att_out, lstm_out], dim=1)
        
        # 输出层
        output = self.output_layer(combined)
        
        return output, attention_map

################################################
# 3. 数据增强
################################################

class AddGaussianNoise:
    """添加高斯噪声"""
    def __init__(self, std=0.1):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

class RandomShift:
    """随机移动信号"""
    def __init__(self, max_shift=10):
        self.max_shift = max_shift
        
    def __call__(self, tensor):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift > 0:
            return torch.cat([tensor[:, :, shift:], torch.zeros_like(tensor[:, :, :shift])], dim=2)
        elif shift < 0:
            return torch.cat([torch.zeros_like(tensor[:, :, :abs(shift)]), tensor[:, :, :shift]], dim=2)
        return tensor

class RandomAmplitudeScale:
    """随机缩放信号幅度"""
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(self, tensor):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return tensor * scale

class Compose:
    """组合多个数据增强方法"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, tensor):
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

################################################
# 4. 训练和评估函数
################################################

def evaluate_model(model, val_loader, device, criterion=nn.BCELoss()):
    """
    评估模型性能
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        criterion: 损失函数
    
    返回:
        val_loss: 验证损失
        val_f1: 验证F1分数
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            # 获取数据
            ecg = batch['ecg'].to(device)
            label = batch['label'].to(device)
            
            # 前向传播
            output, _ = model(ecg)
            
            # 计算损失
            loss = criterion(output, label)
            val_loss += loss.item()
            
            # 计算F1分数
            pred = (output > 0.5).float()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_f1 = compute_f1_score(all_preds, all_labels)
    
    return val_loss, val_f1

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, model_save_path="ecg_peak_detector_best.pth"):
    """
    训练模型
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
        lr: 学习率
        model_save_path: 模型保存路径
    
    返回:
        model: 训练后的模型
        history: 训练历史
    """
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # 初始化训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 计算初始验证损失和F1分数
    print("计算初始验证指标...")
    val_loss, val_f1 = evaluate_model(model, val_loader, device, criterion)
    print(f"初始验证 - Loss: {val_loss:.8f} - F1: {val_f1:.8f}")
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    
    # 记录最佳模型
    best_val_f1 = val_f1
    best_model_state = model.state_dict().copy()
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # 获取数据
            ecg = batch['ecg'].to(device)
            label = batch['label'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output, _ = model(ecg)
            
            # 计算损失
            loss = criterion(output, label)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        val_loss, val_f1 = evaluate_model(model, val_loader, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f} - Val F1: {val_f1:.8f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, model_save_path)
            print(f"保存最佳模型，F1: {best_val_f1:.8f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    print(f"加载最佳模型，F1: {best_val_f1:.8f}")
    
    return model, history

def compute_f1_score(preds, labels):
    """
    计算F1分数
    
    参数:
        preds: 预测值列表
        labels: 真实值列表
    
    返回:
        f1: F1分数
    """
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 计算TP, FP, FN
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    # 计算精确率和召回率
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return f1

################################################
# 5. 预测函数
################################################

def detect_r_peaks(ecg_signal, model, device):
    """
    检测ECG信号中的R-Peaks
    
    参数:
        ecg_signal: 输入的ECG信号，形状为(1250,)
        model: 训练好的模型
        device: 设备（CPU或GPU）
    
    返回:
        r_peaks: R-Peaks的位置索引
    """
    # 确保信号长度为1250
    if len(ecg_signal) != 1250:
        raise ValueError("ECG信号长度必须为1250个点")
    
    # 转换为PyTorch张量并调整形状
    ecg_tensor = torch.from_numpy(ecg_signal).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 1250)
    
    # 模型推理
    with torch.no_grad():
        output, _ = model(ecg_tensor)  # 假设模型返回两个输出，取第一个
    
    # 获取R-Peaks的索引
    r_peaks = (output.squeeze().cpu().numpy() > 0.5).astype(int)  # 转换为二进制数组
    r_peak_indices = np.where(r_peaks == 1)[0]  # 找到R-Peaks的位置索引
    
    return r_peak_indices

def visualize_prediction(signal, true_peaks, pred_peaks, fs=125, save_path="prediction_visualization.png"):
    """
    可视化预测结果
    
    参数:
        signal: ECG信号
        true_peaks: 真实R-Peaks位置
        pred_peaks: 预测R-Peaks位置
        fs: 采样率
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制ECG信号
    time = np.arange(len(signal)) / fs
    plt.plot(time, signal, 'b', label='ECG Signal')
    
    # 绘制真实R-Peaks
    if true_peaks is not None and len(true_peaks) > 0:
        plt.plot(true_peaks / fs, signal[true_peaks], 'ro', label='True R-Peaks')
    
    # 绘制预测R-Peaks
    if pred_peaks is not None and len(pred_peaks) > 0:
        plt.plot(pred_peaks / fs, signal[pred_peaks], 'gx', label='Predicted R-Peaks')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG R-Peaks Detection')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def visualize_r_peaks(ecg_signal, r_peaks, fs=125, save_path="r_peaks_visualization.png"):
    """
    可视化ECG信号及其检测到的R-Peaks
    
    参数:
        ecg_signal: 输入的ECG信号
        r_peaks: 检测到的R-Peaks位置索引
        fs: 采样率
        save_path: 保存路径
    """
    time = np.arange(len(ecg_signal)) / fs  # 计算时间轴
    plt.figure(figsize=(15, 5))
    
    # 绘制ECG信号
    plt.plot(time, ecg_signal, label='ECG Signal', color='blue')
    
    # 绘制R-Peaks
    plt.plot(r_peaks / fs, ecg_signal[r_peaks], 'ro', label='Detected R-Peaks')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Detected R-Peaks')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def visualize_multiple_r_peaks(test_loader, model, device, num_samples=10, save_path="multiple_r_peaks_visualization.png"):
    """
    可视化多个ECG信号及其检测到的R-Peaks
    
    参数:
        test_loader: 测试数据加载器
        model: 训练好的模型
        device: 设备（CPU或GPU）
        num_samples: 要可视化的样本数量
        save_path: 保存路径
    """
    # 创建一个5行2列的子图网格 (对调行列参数)
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    axes = axes.flatten()  # 将2D数组展平为1D，便于遍历
    
    # 获取测试集中的样本
    test_samples = []
    for batch in test_loader:
        # 将批次中的每个样本添加到列表中
        for i in range(batch['ecg'].size(0)):
            test_samples.append({
                'ecg': batch['ecg'][i],
                'label': batch['label'][i] if 'label' in batch else None,
                'r_peaks': batch['r_peaks'][i] if 'r_peaks' in batch else None
            })
        if len(test_samples) >= num_samples:
            break
    
    # 确保我们有足够的样本
    if len(test_samples) < num_samples:
        num_samples = len(test_samples)
    
    # 随机选择样本
    selected_samples = random.sample(test_samples, num_samples)
    
    for i, sample in enumerate(selected_samples):
        if i >= len(axes):
            break
            
        # 获取ECG信号
        ecg = sample['ecg']
        if isinstance(ecg, torch.Tensor):
            if ecg.dim() == 3:  # 如果是(1, 1, length)形状
                ecg_signal = ecg[0, 0].cpu().numpy()
            elif ecg.dim() == 2:  # 如果是(1, length)形状
                ecg_signal = ecg[0].cpu().numpy()
            else:  # 如果是(length)形状
                ecg_signal = ecg.cpu().numpy()
        else:
            print(f"Error: Unexpected ECG type: {type(ecg)}")
            continue
        
        # 获取真实R-Peaks
        if sample['r_peaks'] is not None:
            true_peaks = sample['r_peaks']
            if isinstance(true_peaks, torch.Tensor):
                true_peaks = true_peaks.cpu().numpy()
        else:
            true_peaks = []
        
        # 检测R-Peaks
        try:
            pred_peaks = detect_r_peaks(ecg_signal, model, device)
        except Exception as e:
            print(f"Error detecting R-peaks for sample {i + 1}: {e}")
            continue
        
        # 获取当前子图
        ax = axes[i]
        
        # 绘制ECG信号和R-Peaks
        ax.plot(ecg_signal, label='ECG Signal', color='blue')
        
        # 绘制真实R-Peaks（如果有）
        if len(true_peaks) > 0:
            ax.plot(true_peaks, ecg_signal[true_peaks], 'go', label='True R-Peaks', markersize=6)
        
        # 绘制预测R-Peaks
        if len(pred_peaks) > 0:
            ax.plot(pred_peaks, ecg_signal[pred_peaks], 'ro', label='Predicted R-Peaks', markersize=6)
        
        ax.set_title(f'Sample {i + 1}', fontsize=12)
        ax.set_xlabel('Samples', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        
        # 只在第一个子图显示图例，避免重复
        if i == 0:
            ax.legend(fontsize=9)
        
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围，使所有子图的垂直尺度一致
        y_min = min(ecg_signal) - 0.1
        y_max = max(ecg_signal) + 0.1
        ax.set_ylim(y_min, y_max)
    
    # 调整子图布局，减小子图之间的间距
    plt.tight_layout(pad=2.0)
    
    # 为整个图表添加一个统一标题
    fig.suptitle('ECG Signal with Detected R-Peaks', fontsize=16, y=0.99)
    
    # 保存图像，增加DPI以提高图像质量
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"多张R-Peaks可视化已保存为 {save_path}")################################################
# 6. 数据加载和主函数
################################################

def load_datasets(data_dir):
    """
    加载数据集
    
    参数:
        data_dir: 数据目录
    
    返回:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
    """
    data_dir = Path(data_dir)
    print(f"正在检查数据目录: {data_dir}")
    
    # 加载训练数据集
    train_datasets = []
    for i in range(1, 10):  # 假设有9个训练数据包
        train_file = data_dir / f"training_{i}.h5"
        if train_file.exists():
            print(f"找到训练文件: {train_file}")
            try:
                train_datasets.append(ECGDataset(str(train_file)))
                print(f"成功加载训练文件: {train_file}")
            except Exception as e:
                print(f"加载训练文件 {train_file} 时出错: {e}")
    
    if not train_datasets:
        raise ValueError(f"在 {data_dir} 中找不到任何训练数据文件")
    
    # 合并所有训练数据集
    train_dataset = ConcatDataset(train_datasets)
    
    # 加载验证数据集
    val_file = data_dir / "validation.h5"
    if val_file.exists():
        print(f"找到验证文件: {val_file}")
        try:
            val_dataset = ECGDataset(str(val_file))
            print(f"成功加载验证文件: {val_file}")
        except Exception as e:
            print(f"加载验证文件 {val_file} 时出错: {e}")
            # 如果没有验证文件，从训练集中分割一部分作为验证集
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    else:
        print(f"警告: 找不到验证文件 {val_file}")
        # 如果没有验证文件，从训练集中分割一部分作为验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 加载测试数据集
    test_file = data_dir / "test.h5"
    if test_file.exists():
        print(f"找到测试文件: {test_file}")
        try:
            test_dataset = ECGDataset(str(test_file))
            print(f"成功加载测试文件: {test_file}")
        except Exception as e:
            print(f"加载测试文件 {test_file} 时出错: {e}")
            # 如果没有测试文件，使用验证集作为测试集
            test_dataset = val_dataset
    else:
        print(f"警告: 找不到测试文件 {test_file}")
        # 如果没有测试文件，使用验证集作为测试集
        test_dataset = val_dataset
    
    return train_dataset, val_dataset, test_dataset

# 自定义数据加载器的collate_fn函数，处理可变长度的r_peaks
def custom_collate_fn(batch):
    """
    自定义collate函数，处理可变长度的r_peaks
    
    参数:
        batch: 批次数据
    
    返回:
        collated_batch: 处理后的批次数据
    """
    ecg = torch.stack([item['ecg'] for item in batch])
    label = torch.stack([item['label'] for item in batch])
    
    # 不要尝试堆叠r_peaks，而是保持它们作为列表
    r_peaks = [item['r_peaks'] for item in batch]
    
    return {
        'ecg': ecg,
        'label': label,
        'r_peaks': r_peaks
    }

def main():
    """主函数"""
    # 设置参数
    data_dir = "training_data_VitalDB_quality"  # 数据目录
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    try:
        train_dataset, val_dataset, test_dataset = load_datasets(data_dir)
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        # 列出目录内容以帮助调试
        print(f"目录 {data_dir} 中的文件:")
        for file in Path(data_dir).glob("*"):
            print(f"  {file}")
        return
    
    # 数据增强
    train_transform = Compose([
        AddGaussianNoise(std=0.05),
        RandomShift(max_shift=10),
        RandomAmplitudeScale(min_scale=0.9, max_scale=1.1)
    ])
    
    # 创建数据加载器，使用自定义的collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
    # check_test_dataset(train_loader, set_type="training set")
    # check_test_dataset(val_loader, set_type="validation set")
    # check_test_dataset(test_loader, set_type="testing set")
    # 创建模型
    model = ECGPeakDetector().to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # # 训练模型
    # print("开始训练...")
    # model, history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    
    # # 评估模型
    # print("评估模型...")
    # model.eval()
    # test_loss = 0
    # test_f1 = 0
    # all_preds = []
    # all_labels = []
    
    # with torch.no_grad():
    #     for batch in tqdm(test_loader, desc="测试中"):
    #         ecg = batch['ecg'].to(device)
    #         label = batch['label'].to(device)
            
    #         output, _ = model(ecg)
    #         loss = nn.BCELoss()(output, label)
    #         test_loss += loss.item()
            
    #         # 计算F1分数
    #         pred = (output > 0.5).float()
    #         all_preds.append(pred.cpu().numpy())
    #         all_labels.append(label.cpu().numpy())
    
    # test_loss /= len(test_loader)
    # test_f1 = compute_f1_score(all_preds, all_labels)
    
    # print(f"测试损失: {test_loss:.8f}")
    # print(f"测试F1分数: {test_f1:.8f}")
    
    # # 保存模型
    # torch.save(model.state_dict(), "ecg_peak_detector_final.pth")
    # print("模型已保存为 ecg_peak_detector_final.pth")
    
    # # 可视化训练历史
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['val_loss'], label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(history['val_f1'], label='Val F1')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1 Score')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.savefig('training_history.png')
        # plt.close()
        # 加载模型
    model = load_model("ecg_peak_detector_best.pth", device)    
    # 示例预测
    print("生成示例预测...")
    # 从测试集中获取一个样本
    sample = next(iter(test_loader))
    ecg_signal = sample['ecg'][0, 0].numpy()
    true_peaks = sample['r_peaks'][0]
    
    # 预测R峰
    pred_peaks = detect_r_peaks(ecg_signal, model, device)
    
    # 可视化预测结果
    visualize_prediction(ecg_signal, true_peaks, pred_peaks)
    print("预测可视化已保存为 prediction_visualization.png")

    # # 可视化R-Peaks
    # visualize_r_peaks(ecg_signal, pred_peaks, fs=125, save_path="r_peaks_visualization.png")
    # print("R-Peaks可视化已保存为 r_peaks_visualization.png")

    # 可视化多个样本
    visualize_multiple_r_peaks(test_loader, model, device, num_samples=10, save_path="multiple_r_peaks_visualization.png")

def load_model(model_path, device):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型文件路径
        device: 设备（CPU或GPU）
    
    返回:
        model: 加载的模型
    """
    model = ECGPeakDetector().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式
    return model

def api_detect_r_peaks(ecg_signal, model_path="ecg_peak_detector_final.pth", device=None):
    """
    API函数，检测ECG信号中的R-Peaks
    
    参数:
        ecg_signal: 输入的ECG信号，形状为(1250,)
        model_path: 模型文件路径
        device: 设备（CPU或GPU），如果为None则自动选择
    
    返回:
        r_peaks: R-Peaks的位置索引
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 检测R-Peaks
    r_peaks = detect_r_peaks(ecg_signal, model, device)
    
    return r_peaks

def check_test_dataset(test_loader: DataLoader, set_type: str = "testing set") -> None:
    """
    检查测试数据集中的样本是否重复
    
    参数:
        test_loader: 测试数据加载器
    """
    seen_samples = set()
    for i, sample in tqdm(enumerate(test_loader), desc=f"检查{set_type}数据集"):
        ecg_signal = sample['ecg'][0, 0].numpy()  # 获取ECG信号
        signal_tuple = tuple(ecg_signal)  # 将信号转换为元组以便于哈希
        
        if signal_tuple in seen_samples:
            print(f"重复样本在索引 {i} 处发现")
        else:
            seen_samples.add(signal_tuple)

    print(f"{set_type}中的样本总数: {len(seen_samples)}")

if __name__ == "__main__":
    main() 