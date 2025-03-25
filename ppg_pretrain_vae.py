import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 建立 TensorBoard 日誌
writer = SummaryWriter(log_dir="runs/ppg_vae_power2")

#########################################
# 1. Dataset 定義：使用滑動窗口處理 PPG 資料
#########################################
class SlidingWindowPPGDataset(Dataset):
    """
    從 h5 檔讀取 'ppg' 資料 (N,1250)，使用滑動窗口創建多個 1024 點的片段。
    """
    def __init__(self, h5_path, window_size=1024, stride=512):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.window_size = window_size
        self.stride = stride
        
        with h5py.File(self.h5_path, 'r') as f:
            self.ppg = torch.from_numpy(f['ppg'][:]).float()  # (N,1250)
        
        self.original_N = self.ppg.shape[0]
        self.original_length = self.ppg.shape[1]
        
        # 每個原始信號可產生的窗口數
        self.windows_per_signal = max(1, (self.original_length - self.window_size) // self.stride + 1)
        self.total_windows = self.original_N * self.windows_per_signal
        
        print(f"從 {self.original_N} 個原始信號創建了 {self.total_windows} 個窗口")
        
    def __len__(self):
        return self.total_windows
    
    def __getitem__(self, idx):
        signal_idx = idx // self.windows_per_signal
        window_idx = idx % self.windows_per_signal
        start_pos = window_idx * self.stride
        window = self.ppg[signal_idx, start_pos:start_pos + self.window_size]
        if window.shape[0] < self.window_size:
            padding = self.window_size - window.shape[0]
            window = F.pad(window, (0, padding), "constant", 0)
        window = window.unsqueeze(0)  # (1, window_size)
        return {'signal': window, 'signal_idx': signal_idx}

def create_vae_dataloader(data_dir, batch_size=16, window_size=1024, stride=512, num_workers=16, validation=False):
    """
    從 data_dir 中讀取所有 training_*.h5 檔案，建立一個 ConcatDataset 並返回 DataLoader。
    """
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("training_*.h5"))
    datasets = []
    for f in train_files:
        datasets.append(SlidingWindowPPGDataset(str(f), window_size=window_size, stride=stride))
    if len(datasets) == 0:
        raise FileNotFoundError("找不到任何 training 檔案。")
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    return dataloader

#########################################
# 2. 模型架構：ResBlock Based VAE (處理 1D PPG，1024點)
#########################################
# 2.1 定義 1D ResBlock
class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        """
        單一 ResBlock：經過兩層卷積與 BatchNorm，再做 skip connection
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)

# 2.2 Encoder 模組
class Encoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=16, latent_dim=64):
        """
        改進版 Encoder：初始卷積、三層 ResBlock 與多層下採樣，
        將 1024 點輸入逐步下採樣到 128 點，再透過 global pooling 得到固定 feature vector。
        """
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock1D(base_channels, kernel_size=3)
        
        # 1st downsampling: 1024 -> 512
        self.downsample1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.resblock2 = ResBlock1D(base_channels * 2, kernel_size=3)
        
        # 2nd downsampling: 512 -> 256
        self.downsample2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.resblock3 = ResBlock1D(base_channels * 4, kernel_size=3)
        
        # 3rd downsampling: 256 -> 128
        self.downsample3 = nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)
        self.resblock4 = ResBlock1D(base_channels * 8, kernel_size=3)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 結果 shape: (B, base_channels*8, 1)
        self.fc_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8, latent_dim)
        
    def forward(self, x):
        # x: (B, 1, 1024)
        out = self.relu(self.conv1(x))          # (B, base_channels, 1024)
        out = self.resblock1(out)
        
        out = self.downsample1(out)             # (B, base_channels*2, 512)
        out = self.resblock2(out)
        
        out = self.downsample2(out)             # (B, base_channels*4, 256)
        out = self.resblock3(out)
        
        out = self.downsample3(out)             # (B, base_channels*8, 128)
        out = self.resblock4(out)
        
        pooled = self.global_pool(out).squeeze(-1)  # (B, base_channels*8)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

# 2.3 Decoder 模組
class Decoder(nn.Module):
    def __init__(self, output_length=1024, latent_dim=64, base_channels=16):
        """
        改進版 Decoder：從 latent 向量映射到初始 feature map，再利用多層轉置卷積上採樣回 1024 點。
        """
        super().__init__()
        self.output_length = output_length
        initial_length = output_length // 8  # 1024//8 = 128
        
        self.fc = nn.Linear(latent_dim, base_channels * 8 * initial_length)
        self.relu = nn.ReLU(inplace=True)
        
        # 上採樣 1: 128 -> 256
        self.upconv1 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.resblock1 = ResBlock1D(base_channels * 4, kernel_size=3)
        
        # 上採樣 2: 256 -> 512
        self.upconv2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.resblock2 = ResBlock1D(base_channels * 2, kernel_size=3)
        
        # 上採樣 3: 512 -> 1024
        self.upconv3 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.resblock3 = ResBlock1D(base_channels, kernel_size=3)
        
        self.conv_out = nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)
        
    def forward(self, z):
        # z: (B, latent_dim)
        initial_length = self.output_length // 8  # 128
        out = self.fc(z)                          # (B, base_channels*8 * 128)
        out = self.relu(out)
        out = out.view(z.size(0), -1, initial_length)  # (B, base_channels*8, 128)
        
        out = self.upconv1(out)                   # (B, base_channels*4, 256)
        out = self.resblock1(out)
        
        out = self.upconv2(out)                   # (B, base_channels*2, 512)
        out = self.resblock2(out)
        
        out = self.upconv3(out)                   # (B, base_channels, 1024)
        out = self.resblock3(out)
        
        out = self.conv_out(out)                  # (B, 1, 1024)
        recon = torch.sigmoid(out)                # 限定在 [0,1]
        return recon

# 2.4 VAE 整體模型 (含 reparameterization)
class PPG_VAE(nn.Module):
    def __init__(self, input_length=1024, input_channels=1, latent_dim=64, base_channels=16):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels, base_channels=base_channels, latent_dim=latent_dim)
        self.decoder = Decoder(output_length=input_length, latent_dim=latent_dim, base_channels=base_channels)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        return self.decoder(z)

# 2.5 特征提取與平均 (供下游使用)
class FeatureExtractor:
    """
    從多個窗口提取並平均特徵表示
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(self, dataloader):
        features_by_signal = {}
        for batch in tqdm(dataloader, desc="提取特征"):
            signals = batch['signal'].to(self.device)
            signal_indices = batch['signal_idx'].cpu().numpy()
            mu, logvar = self.model.encoder(signals)
            z = self.model.reparameterize(mu, logvar)
            for i, idx in enumerate(signal_indices):
                features_by_signal.setdefault(idx, []).append(z[i].cpu().numpy())
        features_dict = {idx: np.mean(np.array(f_list), axis=0) for idx, f_list in features_by_signal.items()}
        return features_dict

#########################################
# 2.6 VAE Loss：使用 free bits 與 cyclical beta
#########################################
def vae_loss_free_bits(recon, x, mu, logvar, beta=1.0, free_bits=2.0):
    """
    VAE loss 改進版本：MSE 重建損失 + KL loss（free bits 每個維度最低阈值）
    """
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, latent_dim]
    free_bits_threshold = free_bits / mu.size(1)
    kl_loss = torch.sum(torch.max(kl_per_dim, torch.full_like(kl_per_dim, free_bits_threshold)), dim=1).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def compute_active_units(mu, logvar, threshold=0.01):
    """
    計算活躍潛在單元的數量（平均 KL 大於 threshold）
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    mean_kl_per_dim = kl_per_dim.mean(dim=0)
    active_units = (mean_kl_per_dim > threshold).sum().item()
    return active_units

def get_beta_cyclical(epoch, cycle_length=50, max_beta=0.5):
    """
    周期性 beta 策略：前半周期線性增加，後半周期保持 max_beta
    """
    cycle_position = (epoch % cycle_length) / cycle_length
    if cycle_position < 0.5:
        beta = max_beta * (2.0 * cycle_position)
    else:
        beta = max_beta
    return beta

#########################################
# 3. 訓練與驗證函式，並採用混合精度訓練
#########################################
def preprocess_for_validation(signal, target_length=1024):
    current_length = signal.shape[-1]
    if current_length > target_length:
        start = (current_length - target_length) // 2
        return signal[..., start:start+target_length]
    elif current_length < target_length:
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        return F.pad(signal, (pad_left, pad_right), "constant", 0)
    else:
        return signal

def evaluate_vae(model, dataloader, device='cuda', beta=1.0, free_bits=2.0):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    count = 0
    all_mu, all_logvar = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            if x.shape[-1] != 1024:
                x = preprocess_for_validation(x, 1024)
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss_free_bits(recon, x, mu, logvar, beta=beta, free_bits=free_bits)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            count += batch_size
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    active_units = compute_active_units(all_mu, all_logvar)
    return total_loss / count, total_recon / count, total_kl / count, active_units

# 使用 AMP 加速訓練
def train_vae(model, train_loader, val_loader, optimizer, device='cuda', epochs=20, free_bits=2.0, max_beta=0.5, cycle_length=50, use_amp=False):
    model.to(device)
    
    # 创建保存重建图像的文件夹
    recon_dir = "vae_reconstruction_images"
    os.makedirs(recon_dir, exist_ok=True)
    
    # 创建保存模型的文件夹
    model_dir = "vae_checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存训练历史
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [], 'train_active_units': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [], 'val_active_units': []
    }
    
    # 初始验证 loss (epoch 0)
    print("执行初始验证...")
    init_total, init_recon, init_kl, init_active = evaluate_vae(model, val_loader, device, beta=0.0, free_bits=free_bits)
    print(f"Initial Validation Loss: Total: {init_total:.4f}, Recon: {init_recon:.4f}, KL: {init_kl:.4f}, Active Units: {init_active}/{model.latent_dim}")
    
    # 从验证集获取固定样本用于每个epoch的重建比较
    print("获取固定样本用于可视化...")
    fixed_samples = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['signal'].to(device)
            # 处理长度不匹配问题
            if x.shape[-1] != 1024:
                x = preprocess_for_validation(x, 1024)
            # 取前5个样本
            for i in range(min(5, x.size(0))):
                fixed_samples.append(x[i:i+1])
            if len(fixed_samples) >= 5:
                break
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 混合精度训练设置
    scaler = torch.amp.GradScaler() if use_amp and device == 'cuda' else None
    
    # 用于保存最佳模型
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        # KL annealing: 前 warmup_epochs 逐步增加 beta
        if cycle_length > 0:
            # 循环退火
            beta = max_beta * (1 - np.cos(np.pi * ((epoch-1) % cycle_length) / cycle_length)) / 2
        else:
            # 线性退火
            beta = min(epoch / 10, 1.0) * max_beta
        
        print(f"Epoch {epoch}: beta = {beta:.4f}")
        
        # 添加进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                x = batch['signal'].to(device)
                
                # 打印第一个batch的形状和内存使用情况
                if epoch == 1 and batch_idx == 0:
                    print(f"第一个batch形状: {x.shape}")
                    if device == 'cuda':
                        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                
                # 处理长度不匹配问题
                if x.shape[-1] != 1024:
                    x = preprocess_for_validation(x, 1024)
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        recon, mu, logvar = model(x)
                        loss, recon_loss, kl_loss = vae_loss_free_bits(recon, x, mu, logvar, beta=beta, free_bits=free_bits)
                    
                    # 使用scaler进行反向传播和优化
                    scaler.scale(loss).backward()
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 不使用混合精度训练
                    recon, mu, logvar = model(x)
                    loss, recon_loss, kl_loss = vae_loss_free_bits(recon, x, mu, logvar, beta=beta, free_bits=free_bits)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # 更新进度条信息
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                
                if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_recon = total_recon / (batch_idx + 1)
                    avg_kl = total_kl / (batch_idx + 1)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'recon': f'{avg_recon:.4f}',
                        'kl': f'{avg_kl:.4f}',
                        'beta': f'{beta:.2f}'
                    })
                
                # 第一个epoch的第一个batch特别监控
                if epoch == 1 and batch_idx == 0:
                    print(f"第一个batch训练完成，损失: {loss.item():.6f}")
                    if device == 'cuda':
                        print(f"训练后GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            except Exception as e:
                print(f"处理batch {batch_idx}时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算训练集平均损失
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        
        # 计算训练集活跃单元
        train_active_units = compute_active_units_from_loader(model, train_loader, device, max_samples=1000)
        
        # 保存训练历史
        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        history['train_kl'].append(avg_kl)
        history['train_active_units'].append(train_active_units)
        
        # 记录到TensorBoard
        writer.add_scalar('Train/TotalLoss', avg_loss, epoch)
        writer.add_scalar('Train/ReconstructionLoss', avg_recon, epoch)
        writer.add_scalar('Train/KLLoss', avg_kl, epoch)
        writer.add_scalar('Train/Beta', beta, epoch)
        writer.add_scalar('Train/ActiveUnits', train_active_units, epoch)
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}), Active Units: {train_active_units}/{model.latent_dim}, beta: {beta:.2f}")
        
        # 验证
        val_total, val_recon, val_kl, val_active = evaluate_vae(model, val_loader, device, beta=beta, free_bits=free_bits)
        history['val_loss'].append(val_total)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        history['val_active_units'].append(val_active)
        
        writer.add_scalar('Val/TotalLoss', val_total, epoch)
        writer.add_scalar('Val/ReconstructionLoss', val_recon, epoch)
        writer.add_scalar('Val/KLLoss', val_kl, epoch)
        writer.add_scalar('Val/ActiveUnits', val_active, epoch)
        
        print(f"Epoch {epoch}: Val Loss: Total: {val_total:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Active Units: {val_active}/{model.latent_dim}")
        
        # 学习率调度
        scheduler.step(val_total)
        
        # 保存最佳模型
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'history': history,
                'active_units': val_active
            }, os.path.join(model_dir, 'best_model.pth'))
            print(f"保存最佳模型 (Epoch {epoch}, Loss: {best_val_loss:.6f})")
        
        # 使用线程在后台保存重建图像
        try:
            # 简化可视化过程，直接在主线程中执行
            if epoch % 5 == 0 or epoch == 1:  # 每5个epoch保存一次
                save_reconstruction_image(epoch, model, fixed_samples, recon_dir, beta)
        except Exception as e:
            print(f"保存重建图像时出错: {e}")
        
        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_total,
                'history': history,
                'active_units': val_active
            }, os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"训练完成。最佳模型在 Epoch {best_epoch}，验证损失: {best_val_loss:.6f}")
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, history

# 辅助函数：计算活跃单元
def compute_active_units_from_loader(model, dataloader, device, max_samples=1000):
    """从数据加载器计算活跃单元，限制样本数量以提高效率"""
    model.eval()
    all_mu = []
    all_logvar = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            # 处理长度不匹配问题
            if x.shape[-1] != 1024:
                x = preprocess_for_validation(x, 1024)
            _, mu, logvar = model(x)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            
            sample_count += x.size(0)
            if sample_count >= max_samples:
                break
    
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    return compute_active_units(all_mu, all_logvar)

# 简化的保存重建图像函数
def save_reconstruction_image(epoch, model, samples, save_dir, beta):
    """简化的重建图像保存函数，直接在主线程中执行"""
    fig = plt.figure(figsize=(15, 10))
    
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(samples):
            if i >= 5:  # 最多显示5个样本
                break
                
            # 获取重建结果
            recon, mu, logvar = model(x)
            
            # 提取数据
            x_sample = x[0].cpu().numpy().squeeze()
            recon_sample = recon[0].cpu().numpy().squeeze()
            
            # 创建子图
            ax1 = fig.add_subplot(5, 2, i*2+1)
            ax2 = fig.add_subplot(5, 2, i*2+2)
            
            # 绘制原始信号
            ax1.plot(x_sample)
            ax1.set_title(f"样本 {i+1} - 原始PPG信号")
            ax1.set_ylabel("振幅")
            
            # 绘制重建信号
            ax2.plot(recon_sample, color='red')
            ax2.set_title(f"样本 {i+1} - 重建PPG信号")
            ax2.set_ylabel("振幅")
    
    plt.suptitle(f"Epoch {epoch} 重建结果 (beta={beta:.2f})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, f"recon_epoch_{epoch:03d}.png"))
    plt.close(fig)
    print(f"已保存第 {epoch} 轮重建图像")

#########################################
# 4. 特徵提取與可視化
#########################################
def extract_and_visualize_features(model, dataloader, device='cuda', n_samples=5):
    feature_extractor = FeatureExtractor(model, device)
    features_dict = feature_extractor.extract_features(dataloader)
    print(f"共提取了 {len(features_dict)} 個信號的特征表示")
    sample_indices = list(features_dict.keys())[:n_samples]
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(sample_indices):
        plt.subplot(n_samples, 1, i+1)
        plt.bar(range(len(features_dict[idx])), features_dict[idx])
        plt.title(f"信號 {idx} 的特征表示")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Value")
        plt.tight_layout()
    plt.tight_layout()
    plt.show()
    return features_dict

def visualize_reconstruction(model, dataloader, device='cuda'):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            recon, _, _ = model(x)
            x_sample = x[0].cpu().numpy().squeeze()
            recon_sample = recon[0].cpu().numpy().squeeze()
            break
    plt.figure(figsize=(12,4))
    plt.plot(x_sample, label='Input PPG')
    plt.plot(recon_sample, label='Reconstruction', linestyle='--')
    plt.title("PPG VAE 重建結果")
    plt.xlabel("Time")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.show()

def visualize_latent_space(model, dataloader, device='cuda'):
    model.eval()
    all_mu, all_logvar = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            _, mu, logvar = model(x)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    kl_per_dim = -0.5 * (1 + all_logvar - np.square(all_mu) - np.exp(all_logvar))
    mean_kl_per_dim = np.mean(kl_per_dim, axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_kl_per_dim)), mean_kl_per_dim)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Mean KL Divergence')
    plt.title('KL Divergence per Latent Dimension')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Active Unit Threshold')
    plt.legend()
    plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.hist(all_mu.flatten(), bins=50, alpha=0.7)
    ax1.set_title('Distribution of Latent Means')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    all_std = np.exp(0.5 * all_logvar)
    ax2.hist(all_std.flatten(), bins=50, alpha=0.7)
    ax2.set_title('Distribution of Latent Standard Deviations')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

#########################################
# 5. 主程式：建立 DataLoader、訓練 VAE 並儲存模型
#########################################
def main_vae():
    data_dir = "training_data_VitalDB_quality"
    window_size = 1024
    stride = 512
    
    # 减小批次大小
    batch_size = 64  # 从1024减小到64
    
    # 减少工作线程数
    num_workers = 2  # 从4减少到2
    
    train_loader = create_vae_dataloader(data_dir, batch_size=batch_size, window_size=window_size, stride=stride, num_workers=num_workers)
    val_loader = create_vae_dataloader(data_dir, batch_size=batch_size, window_size=window_size, stride=stride, num_workers=num_workers, validation=True)
    
    print(f"Train: {len(train_loader.dataset)} 個窗口, 共 {len(train_loader)} 個 batch")
    print(f"Validation: {len(val_loader.dataset)} 個窗口, 共 {len(val_loader)} 個 batch")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 打印GPU信息
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體總量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"當前GPU記憶體使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 创建模型
    model = PPG_VAE(input_length=window_size, input_channels=1, latent_dim=32, base_channels=32)
    print("模型參數總數:", sum(p.numel() for p in model.parameters()))
    
    # 使用较小的学习率
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("開始訓練 VAE (改進版)...")
    
    # 禁用混合精度训练
    model, history = train_vae(model, train_loader, val_loader, optimizer, device=device, epochs=20, free_bits=2.0, max_beta=0.1, cycle_length=50, use_amp=False)
    
    print("顯示重建結果...")
    visualize_reconstruction(model, val_loader, device=device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, "ppg_vae_improved.pth")
    print("儲存模型到 ppg_vae_improved.pth")
    
    return model, history

if __name__ == "__main__":
    main_vae()
