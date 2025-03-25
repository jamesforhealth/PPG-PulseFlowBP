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
from collections import Counter
import seaborn as sns

# 設定 cudnn.benchmark，對於固定尺寸的輸入有幫助
torch.backends.cudnn.benchmark = True

# 建立 TensorBoard 日誌
writer = SummaryWriter(log_dir="runs/ppg_vqvae_optimized")

#########################################
# 1. Dataset 定義：使用滑動窗口處理 PPG 資料
#########################################
class SlidingWindowPPGDataset(Dataset):
    """
    從 h5 檔讀取 'ppg' 資料 (N,1250)，使用滑動窗口創建多個 1024 點的片段。
    若存在 'bp' 標籤，則同時返回（可用於條件學習）。
    """
    def __init__(self, h5_path, window_size=1024, stride=512):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.window_size = window_size
        self.stride = stride
        
        with h5py.File(self.h5_path, 'r') as f:
            ppg_data = f['ppg'][:]
            ppg_min, ppg_max = np.min(ppg_data), np.max(ppg_data)
            if ppg_max > 1.0 or ppg_min < 0.0:
                print(f"警告: 原始數據範圍為 [{ppg_min}, {ppg_max}]，進行規範化")
                ppg_data = (ppg_data - ppg_min) / (ppg_max - ppg_min + 1e-8)
            self.ppg = torch.from_numpy(ppg_data).float()  # (N,1250)
            self.bp = torch.from_numpy(f['bp'][:]).float() if 'bp' in f else None
        
        self.original_N = self.ppg.shape[0]
        self.original_length = self.ppg.shape[1]
        self.windows_per_signal = max(1, (self.original_length - self.window_size) // self.stride + 1)
        self.total_windows = self.original_N * self.windows_per_signal
        print(f"從 {self.original_N} 個原始信號創建了 {self.total_windows} 個窗口")
        
    def __len__(self):
        return self.total_windows
    
    def __getitem__(self, idx):
        signal_idx = idx // self.windows_per_signal
        window_idx = idx % self.windows_per_signal
        start_pos = window_idx * self.stride
        window = self.ppg[signal_idx, start_pos:start_pos+self.window_size]
        if window.shape[0] < self.window_size:
            padding = self.window_size - window.shape[0]
            window = F.pad(window, (0, padding), "constant", 0)
        window = torch.clamp(window, 0.0, 1.0)
        window = window.unsqueeze(0)  # (1, window_size)
        sample = {'signal': window, 'signal_idx': signal_idx}
        if self.bp is not None:
            sample['bp'] = self.bp[signal_idx]
        return sample

def create_vae_dataloader(data_dir, batch_size=128, window_size=1024, stride=512, num_workers=16):
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("training_*.h5"))
    datasets = [SlidingWindowPPGDataset(str(f), window_size, stride) for f in train_files]
    if len(datasets) == 0:
        raise FileNotFoundError("找不到任何 training 檔案。")
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=4)
    return dataloader

#########################################
# 2. 模型架構：Conditional VQ-VAE
#########################################
# 2.1 定義 1D ResBlock
class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
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
    def __init__(self, input_channels=1, base_channels=16, embedding_dim=64):
        """
        將 1024 點 PPG 信號編碼到 (B, embedding_dim, 128)
        """
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock1D(base_channels, kernel_size=3)
        self.downsample1 = nn.Conv1d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)  # 1024 -> 512
        self.resblock2 = ResBlock1D(base_channels*2, kernel_size=3)
        self.downsample2 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)  # 512 -> 256
        self.resblock3 = ResBlock1D(base_channels*4, kernel_size=3)
        self.downsample3 = nn.Conv1d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self.resblock4 = ResBlock1D(base_channels*8, kernel_size=3)
        self.conv_out = nn.Conv1d(base_channels*8, embedding_dim, kernel_size=1)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.resblock1(out)
        out = self.downsample1(out)
        out = self.resblock2(out)
        out = self.downsample2(out)
        out = self.resblock3(out)
        out = self.downsample3(out)
        out = self.resblock4(out)
        z_e = self.conv_out(out)  # (B, embedding_dim, 128)
        return z_e

# 2.3 Vector Quantizer 模組（改進 EMA 更新）
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, decay=0.95, epsilon=1e-3):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('_ema_cluster_size', torch.ones(num_embeddings) * 0.1)
        self._ema_w = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.1)
        self._ema_w.requires_grad = False
        self._decay = decay
        self._epsilon = epsilon
        self.initial_training_done = False
    def forward(self, z_e):
        # z_e: (B, embedding_dim, T)
        B, D, T = z_e.shape
        z_e_flat = z_e.permute(0,2,1).contiguous().view(-1, self.embedding_dim)
        z_e_norm = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
        embedding_norm = torch.sum(self.embedding.weight ** 2, dim=1)
        d = z_e_norm + embedding_norm - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        d = torch.clamp(d, min=0.0)
        min_encoding_indices = torch.argmin(d, dim=1)
        encodings = F.one_hot(min_encoding_indices, self.num_embeddings).float()
        z_q_flat = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q_flat.view(B, T, self.embedding_dim).permute(0,2,1).contiguous()
        e_sum = torch.sum(encodings, dim=0)
        total_count = torch.sum(e_sum)
        e_mean = e_sum / (total_count + 1e-8)
        entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-8))
        perplexity = torch.exp(entropy)
        vq_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        loss = vq_loss + self.commitment_cost * commitment_loss
        if self.training:
            n = torch.sum(encodings, dim=0)
            if torch.sum(n) > 0 or self.initial_training_done:
                self.initial_training_done = True
                self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (1 - self._decay) * (n + self._epsilon)
                z_e_sum = torch.matmul(encodings.t(), z_e_flat)
                self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * z_e_sum
                n = torch.clamp(self._ema_cluster_size, min=2.0)
                embedding_normalized = self._ema_w / n.unsqueeze(1)
                self.embedding.weight.data = embedding_normalized
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, loss, perplexity, min_encoding_indices.view(B, T)

# 2.4 Decoder 模組
class Decoder(nn.Module):
    def __init__(self, output_length=1024, embedding_dim=64, base_channels=16):
        super().__init__()
        self.output_length = output_length
        initial_length = output_length // 8  # 1024//8 = 128
        self.conv_in = nn.Conv1d(embedding_dim, base_channels*8, kernel_size=1)
        self.resblock0 = ResBlock1D(base_channels*8, kernel_size=3)
        self.upconv1 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1)
        self.resblock1 = ResBlock1D(base_channels*4, kernel_size=3)
        self.upconv2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.resblock2 = ResBlock1D(base_channels*2, kernel_size=3)
        self.upconv3 = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)
        self.resblock3 = ResBlock1D(base_channels, kernel_size=3)
        self.conv_out = nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)
    def forward(self, z_q):
        out = self.conv_in(z_q)
        out = self.resblock0(out)
        out = self.upconv1(out)
        out = self.resblock1(out)
        out = self.upconv2(out)
        out = self.resblock2(out)
        out = self.upconv3(out)
        out = self.resblock3(out)
        out = self.conv_out(out)
        if out.shape[2] != self.output_length:
            out = F.interpolate(out, size=self.output_length, mode='linear', align_corners=True)
        recon = torch.sigmoid(out)
        return recon

# 2.5 Conditional VQ-VAE 模型
class PPG_VQVAE(nn.Module):
    def __init__(self, input_length=1024, input_channels=1, embedding_dim=64, num_embeddings=128, 
                 commitment_cost=0.5, base_channels=16, use_condition=False, condition_dim=0, aux_loss_weight=0.1):
        """
        若 use_condition 為 True，則額外接收 condition（例如血壓校正標籤），並通過輔助回歸頭生成輔助損失。
        """
        super().__init__()
        self.use_condition = use_condition
        self.condition_dim = condition_dim
        self.aux_loss_weight = aux_loss_weight
        self.encoder = Encoder(input_channels=input_channels, base_channels=base_channels, embedding_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(output_length=input_length, embedding_dim=embedding_dim, base_channels=base_channels)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings  # 解決屬性缺失問題
        if self.use_condition:
            self.aux_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim//2),
                nn.ReLU(),
                nn.Linear(embedding_dim//2, condition_dim)
            )
    def forward(self, x, condition=None):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, encoding_indices = self.vq(z_e)
        recon = self.decoder(z_q)
        output = {
            'reconstruction': recon,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encoding_indices': encoding_indices,
            'z_e': z_e,
            'z_q': z_q
        }
        if self.use_condition and condition is not None:
            z_q_avg = torch.mean(z_q, dim=2)
            bp_pred = self.aux_head(z_q_avg)
            output['bp_pred'] = bp_pred
        return output
    def encode(self, x):
        z_e = self.encoder(x)
        z_q, _, _, encoding_indices = self.vq(z_e)
        return z_q, encoding_indices
    def decode(self, z_q):
        return self.decoder(z_q)
    def encode_indices(self, x):
        z_e = self.encoder(x)
        _, _, _, encoding_indices = self.vq(z_e)
        return encoding_indices

#########################################
# 2.6 損失函數
#########################################
def vqvae_loss(outputs, x, beta=1.0):
    recon = outputs['reconstruction']
    vq_loss = outputs['vq_loss']
    if recon.shape != x.shape:
        recon = F.interpolate(recon, size=x.shape[2], mode='linear', align_corners=True)
    recon_loss = F.mse_loss(recon, x)
    total_loss = beta * recon_loss + vq_loss
    if torch.isnan(total_loss):
        print("警告: 總損失為 NaN！")
        return torch.tensor(1.0, device=x.device, requires_grad=True), recon_loss, vq_loss
    return total_loss, recon_loss, vq_loss

def conditional_aux_loss(bp_pred, bp_target):
    return F.l1_loss(bp_pred, bp_target)

#########################################
# 3. 訓練與驗證函式（混合精度）
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

def evaluate_vqvae(model, dataloader, device='cuda', beta=1.0):
    model.eval()
    total_loss = total_recon_loss = total_vq_loss = total_perplexity = 0.0
    count = 0
    all_indices = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            if x.shape[-1] != 1024:
                x = preprocess_for_validation(x, 1024)
            outputs = model(x)
            loss, recon_loss, vq_loss = vqvae_loss(outputs, x, beta=beta)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_vq_loss += vq_loss.item() * batch_size
            perp = outputs['perplexity'].item()
            total_perplexity += perp * batch_size
            count += batch_size
            all_indices.append(outputs['encoding_indices'].cpu().numpy().flatten())
    avg_loss = total_loss / count
    avg_recon_loss = total_recon_loss / count
    avg_vq_loss = total_vq_loss / count
    avg_perplexity = total_perplexity / count
    all_indices = np.concatenate(all_indices)
    codebook_usage = len(np.unique(all_indices)) / model.num_embeddings * 100.0
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
        'perplexity': avg_perplexity,
        'codebook_usage': codebook_usage,
        'indices': all_indices
    }

def train_vqvae(model, train_loader, val_loader, optimizer, device='cuda', epochs=500, beta=1.0):
    model.to(device)
    scaler = torch.amp.GradScaler()  # 使用新版 API
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    history = {
        'train_loss': [], 'train_recon_loss': [], 'train_vq_loss': [],
        'train_perplexity': [], 'train_codebook_usage': [],
        'val_loss': [], 'val_recon_loss': [], 'val_vq_loss': [],
        'val_perplexity': [], 'val_codebook_usage': [], 'learning_rates': []
    }
    print("初始評估...")
    val_metrics = evaluate_vqvae(model, val_loader, device, beta=beta)
    print(f"Initial Validation: Loss: {val_metrics['loss']:.8f}, Recon: {val_metrics['recon_loss']:.8f}, "
          f"VQ: {val_metrics['vq_loss']:.8f}, Perplexity: {val_metrics['perplexity']:.2f}, "
          f"Codebook Usage: {val_metrics['codebook_usage']:.2f}%")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = total_recon_loss = total_vq_loss = total_perplexity = 0.0
        all_train_indices = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch['signal'].to(device)
            if x.shape[-1] != 1024:
                x = preprocess_for_validation(x, 1024)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(x)
                loss, recon_loss, vq_loss = vqvae_loss(outputs, x, beta=beta)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            perp = outputs['perplexity'].item()
            total_perplexity += perp
            all_train_indices.append(outputs['encoding_indices'].detach().cpu().numpy().flatten())
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_vq_loss = total_vq_loss / len(train_loader)
        avg_perplexity = total_perplexity / len(train_loader)
        all_train_indices = np.concatenate(all_train_indices)
        train_codebook_usage = len(np.unique(all_train_indices)) / model.num_embeddings * 100.0
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        history['train_loss'].append(avg_loss)
        history['train_recon_loss'].append(avg_recon_loss)
        history['train_vq_loss'].append(avg_vq_loss)
        history['train_perplexity'].append(avg_perplexity)
        history['train_codebook_usage'].append(train_codebook_usage)
        writer.add_scalar('Train/TotalLoss', avg_loss, epoch)
        writer.add_scalar('Train/ReconstructionLoss', avg_recon_loss, epoch)
        writer.add_scalar('Train/VQLoss', avg_vq_loss, epoch)
        writer.add_scalar('Train/Perplexity', avg_perplexity, epoch)
        writer.add_scalar('Train/CodebookUsage', train_codebook_usage, epoch)
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.8f} (Recon: {avg_recon_loss:.8f}, VQ: {avg_vq_loss:.8f}), "
              f"Perplexity: {avg_perplexity:.2f}, Codebook Usage: {train_codebook_usage:.2f}%, LR: {current_lr:.6f}")
        val_metrics = evaluate_vqvae(model, val_loader, device, beta=beta)
        history['val_loss'].append(val_metrics['loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['val_vq_loss'].append(val_metrics['vq_loss'])
        history['val_perplexity'].append(val_metrics['perplexity'])
        history['val_codebook_usage'].append(val_metrics['codebook_usage'])
        writer.add_scalar('Val/TotalLoss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/ReconstructionLoss', val_metrics['recon_loss'], epoch)
        writer.add_scalar('Val/VQLoss', val_metrics['vq_loss'], epoch)
        writer.add_scalar('Val/Perplexity', val_metrics['perplexity'], epoch)
        writer.add_scalar('Val/CodebookUsage', val_metrics['codebook_usage'], epoch)
        print(f"Epoch {epoch}: Val Loss: {val_metrics['loss']:.8f} (Recon: {val_metrics['recon_loss']:.8f}, VQ: {val_metrics['vq_loss']:.8f}), "
              f"Perplexity: {val_metrics['perplexity']:.2f}, Codebook Usage: {val_metrics['codebook_usage']:.2f}%")
        scheduler.step(val_metrics['loss'])
        if epoch % 5 == 0 or epoch == 1:
            fig_recon = visualize_reconstructions(model, val_loader, epoch, device, n_samples=5, save_to_tensorboard=True)
            fig_cb, fig_heat = visualize_codebook_usage(val_metrics['indices'], model.num_embeddings, epoch, save_to_tensorboard=True)
            visualize_latent_space_epoch(model, val_loader, epoch, device, save_to_tensorboard=True)
            if epoch % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }, f"vqvae_checkpoint_epoch_{epoch}.pth")
    return model, history

#########################################
# 4. 評估與可視化函式
#########################################
def visualize_reconstructions(model, dataloader, epoch, device='cuda', n_samples=5, save_to_tensorboard=False):
    model.eval()
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            outputs = model(x)
            recon = outputs['reconstruction']
            for i in range(min(n_samples, x.size(0))):
                samples.append({'original': x[i,0].cpu().numpy(), 'recon': recon[i,0].cpu().numpy()})
            if len(samples) >= n_samples:
                break
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2*n_samples))
    for i, sample in enumerate(samples):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(sample['original'], label='Original')
        ax.plot(sample['recon'], label='Reconstructed', alpha=0.8)
        ax.set_title(f"Sample {i+1}")
        ax.set_ylim(0,1)
        ax.legend()
    plt.tight_layout()
    if save_to_tensorboard:
        writer.add_figure("Reconstructions", fig, epoch)
        plt.close(fig)
    else:
        plt.show()
    return fig

def visualize_codebook_usage(indices, num_embeddings, epoch, save_to_tensorboard=False):
    counter = Counter(indices)
    counts = [counter.get(i,0) for i in range(num_embeddings)]
    usage_ratio = len(counter)/num_embeddings*100.0
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(range(num_embeddings), counts)
    ax.set_xlabel('Codebook Index')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Codebook Usage (Epoch {epoch}): {len(counter)}/{num_embeddings} used ({usage_ratio:.2f}%)')
    if save_to_tensorboard:
        writer.add_figure("CodebookUsage", fig, epoch)
        plt.close(fig)
    else:
        plt.show()
    fig2, ax2 = plt.subplots(figsize=(12,3))
    heatmap_data = np.log1p(np.array(counts))
    sns.heatmap(heatmap_data.reshape(1,-1), ax=ax2, cmap='viridis', cbar_kws={'label':'log(Freq+1)'})
    ax2.set_title(f'Codebook Heatmap (Epoch {epoch})')
    ax2.set_xlabel('Codebook Index')
    ax2.set_yticks([])
    if save_to_tensorboard:
        writer.add_figure("CodebookHeatmap", fig2, epoch)
        plt.close(fig2)
    else:
        plt.show()
    return fig, fig2

def visualize_latent_space_epoch(model, dataloader, epoch, device='cuda', n_samples=10, save_to_tensorboard=False):
    model.eval()
    all_mu = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            z_e = model.encoder(x)
            all_mu.append(z_e[:,:,z_e.shape[2]//2].cpu().numpy())
            if len(all_mu)*x.size(0) >= n_samples:
                break
    all_mu = np.concatenate(all_mu, axis=0)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(all_mu[:,0], all_mu[:,1], alpha=0.5, s=5)
    ax.set_title(f'2D Projection of Latent Space (Epoch {epoch})')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True)
    if save_to_tensorboard:
        writer.add_figure("LatentSpaceProjection", fig, epoch)
        plt.close(fig)
    else:
        plt.show()
    return fig

def visualize_latent_space(model, dataloader, device='cuda'):
    model.eval()
    all_mu = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            z_e = model.encoder(x)
            all_mu.append(z_e[:,:,z_e.shape[2]//2].cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)
    plt.figure(figsize=(12,6))
    plt.hist(all_mu.flatten(), bins=50, alpha=0.7)
    plt.title('Distribution of Latent Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

#########################################
# 5. 主程式：建立 DataLoader、訓練並儲存模型
#########################################
def main_vqvae():
    data_dir = "training_data_VitalDB_quality"  # 根據實際路徑調整
    window_size = 1024
    stride = 512
    train_loader = create_vae_dataloader(data_dir, batch_size=128, window_size=window_size, stride=stride, num_workers=16)
    val_dataset = SlidingWindowPPGDataset(os.path.join(data_dir, "validation.h5"), window_size=window_size, stride=window_size)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=16, pin_memory=True, prefetch_factor=4)
    print(f"Train windows: {len(train_loader.dataset)}, Val windows: {len(val_loader.dataset)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = PPG_VQVAE(
        input_length=window_size,
        input_channels=1,
        embedding_dim=64,
        num_embeddings=128,
        commitment_cost=0.5,
        base_channels=16,
        use_condition=False,
        condition_dim=0,
        aux_loss_weight=0.1
    )
    print("模型參數總數:", sum(p.numel() for p in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    model, history = train_vqvae(model, train_loader, val_loader, optimizer, device=device, epochs=50, beta=1.0)
    eval_results = evaluate_vqvae(model, val_loader, device=device, beta=1.0)
    print("最終評估指標:")
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.8f}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'eval_metrics': {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
    }, "ppg_vqvae_final.pth")
    print("儲存最終模型到 ppg_vqvae_final.pth")
    return model, history, eval_results

if __name__ == "__main__":
    main_vqvae()
