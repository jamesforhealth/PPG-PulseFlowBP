import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import h5py
from pathlib import Path

class SlidingWindowPPGDataset(Dataset):
    """使用滑動窗口創建自監督學習的PPG數據集"""
    def __init__(self, h5_path, window_size=1250, stride=125, transforms=None):
        super().__init__()
        self.h5_path = h5_path if isinstance(h5_path, Path) else Path(h5_path)
        self.window_size = window_size
        self.stride = stride
        self.transforms = transforms
        
        # 載入原始PPG數據
        with h5py.File(self.h5_path, 'r') as f:
            self.raw_ppg = torch.from_numpy(f['ppg'][:]).float()  # (N, signal_length)
        
        # 預處理：計算有多少個窗口
        self.indices = []
        for i in range(len(self.raw_ppg)):
            signal_length = self.raw_ppg[i].shape[0]
            # 為每個原始PPG序列生成窗口索引
            for start in range(0, signal_length - window_size + 1, stride):
                self.indices.append((i, start))
        
        print(f"數據集 {self.h5_path.name} 創建了 {len(self.indices)} 個窗口樣本")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx, start = self.indices[idx]
        
        # 提取源窗口
        source_window = self.raw_ppg[sample_idx][start:start+self.window_size]
        source_window = source_window.unsqueeze(0)  # 添加通道維度: (1, window_size)
        
        # 為相同窗口創建不同增強版本作為正樣本
        if self.transforms:
            anchor = self.transforms(source_window.clone())
            positive = self.transforms(source_window.clone())
        else:
            anchor = source_window
            # 簡單的時間偏移作為增強
            shift = np.random.randint(1, 10)
            positive = torch.roll(source_window, shifts=shift, dims=1)
        
        # 創建負樣本 - 從不同位置或不同患者取樣
        neg_sample_idx = sample_idx
        while neg_sample_idx == sample_idx:
            neg_sample_idx = np.random.randint(0, len(self.raw_ppg))
        
        neg_start = np.random.randint(0, self.raw_ppg[neg_sample_idx].shape[0] - self.window_size + 1)
        negative = self.raw_ppg[neg_sample_idx][neg_start:neg_start+self.window_size]
        negative = negative.unsqueeze(0)  # (1, window_size)
        
        # 創建相鄰窗口作為時間連續性樣本
        if start + self.stride + self.window_size <= self.raw_ppg[sample_idx].shape[0]:
            temporal_window = self.raw_ppg[sample_idx][start+self.stride:start+self.stride+self.window_size]
            temporal_window = temporal_window.unsqueeze(0)  # (1, window_size)
        else:
            # 如果已經到序列末尾，使用稍微偏移的窗口
            temporal_window = torch.roll(source_window, shifts=self.stride//2, dims=1)
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'temporal': temporal_window,
            'sample_idx': sample_idx
        }


# PPG數據增強類
class PPGTransforms:
    """PPG信號的數據增強方法"""
    def __init__(self, gaussian_noise_std=0.01, scaling_range=(0.95, 1.05)):
        self.gaussian_noise_std = gaussian_noise_std
        self.scaling_range = scaling_range
    
    def __call__(self, signal):
        # 添加高斯噪聲
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(signal) * self.gaussian_noise_std
            signal = signal + noise
        
        # 縮放
        if self.scaling_range != (1.0, 1.0):
            scale = torch.FloatTensor(1).uniform_(*self.scaling_range)
            signal = signal * scale
        
        return signal

# 自監督學習的損失函數
class TemporalConsistencyLoss(nn.Module):
    """時間一致性損失：確保相鄰窗口的表示相似"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor_features, temporal_features):
        # 歸一化特徵，計算餘弦相似度
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        temporal_features = F.normalize(temporal_features, p=2, dim=1)
        
        # 計算相似度矩陣
        similarity = torch.mm(anchor_features, temporal_features.t()) / self.temperature
        
        # 對角線是正樣本對
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # 交叉熵損失
        loss = F.cross_entropy(similarity, labels)
        return loss

class ContrastiveLoss(nn.Module):
    """對比損失：使正樣本對接近，負樣本對遠離"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor_features, positive_features, negative_features):
        # 歸一化特徵
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        positive_features = F.normalize(positive_features, p=2, dim=1)
        negative_features = F.normalize(negative_features, p=2, dim=1)
        
        # 計算正樣本對的相似度
        positive_similarity = torch.sum(anchor_features * positive_features, dim=1)
        # 計算負樣本對的相似度
        negative_similarity = torch.sum(anchor_features * negative_features, dim=1)
        
        # InfoNCE損失
        logits = torch.cat([
            positive_similarity.unsqueeze(1) / self.temperature,
            negative_similarity.unsqueeze(1) / self.temperature
        ], dim=1)
        
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

# 預訓練模型架構
class PPGPretrainModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        # 使用現有的PPG編碼器作為基礎
        self.encoder = base_encoder
        
        # 添加投影頭，用於自監督學習
        self.projection_head = nn.Sequential(
            nn.Linear(base_encoder.fc_latent[-2].out_features, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        # 編碼器特徵
        features = self.encoder(x)
        # 投影特徵用於對比學習
        projected = self.projection_head(features)
        
        return features, projected


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 1.2 残差块
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + residual
        return self.act(x)

# 1.3 下采样块
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool1d(2),
            ConvBlock1D(in_ch, out_ch),
            ResBlock1D(out_ch)
        )
    
    def forward(self, x):
        return self.conv(x)

# 1.4 上采样块 (修复通道数不匹配问题)
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        # 注意：输入通道数是in_ch + skip_ch，因为会和skip连接
        self.conv = nn.Sequential(
            ConvBlock1D(in_ch * 2, out_ch),  # 修改这里以确保通道数匹配
            ResBlock1D(out_ch)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # 处理奇数长度情况
        if x.size(2) != skip.size(2):
            x = F.pad(x, (0, skip.size(2) - x.size(2)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# 1.5 TCN块
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.dilated_convs = nn.ModuleList()
        
        for d in dilations:
            self.dilated_convs.append(nn.Sequential(
                nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(in_ch),
                nn.ReLU(inplace=True)
            ))
        
        self.output_conv = nn.Sequential(
            nn.Conv1d(in_ch * len(dilations), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        outputs = []
        for dilated_conv in self.dilated_convs:
            outputs.append(dilated_conv(x))
        
        out = torch.cat(outputs, dim=1)
        return self.output_conv(out)


class PPGEncoder(nn.Module):
    def __init__(self, in_channels=1, base_ch=8, latent_dim=128):
        super().__init__()
        
        # 初始卷积层
        self.enc_conv1 = nn.Sequential(
            ConvBlock1D(in_channels, base_ch),
            ResBlock1D(base_ch)
        )
        
        # Encoder路径（下采样）
        self.down1 = DownBlock(base_ch, base_ch*2)  # -> 625
        self.down2 = DownBlock(base_ch*2, base_ch*4)  # -> 312
        self.down3 = DownBlock(base_ch*4, base_ch*8)  # -> 156
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock1D(base_ch*8, base_ch*16),
            ResBlock1D(base_ch*16),
            ConvBlock1D(base_ch*16, base_ch*8)
        )
        
        # 最终池化和线性层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Sequential(
            nn.Linear(base_ch*8, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 输入: [B, 1, 1250]
        # 编码器路径
        c1 = self.enc_conv1(x)          # [B, 16, 1250]
        c2 = self.down1(c1)             # [B, 32, 625]
        c3 = self.down2(c2)             # [B, 64, 312]
        c4 = self.down3(c3)             # [B, 128, 156]
        
        # 瓶颈
        b = self.bottleneck(c4)         # [B, 128, 156]
        
        # 全局池化
        pooled = self.global_pool(b).squeeze(-1)  # [B, 128]
        
        # 潜在空间表示
        latent = self.fc_latent(pooled)  # [B, latent_dim]
        
        return latent

# 1.7 个人信息和血管属性编码器 (简化版)
class PersonalVascularEncoder(nn.Module):
    def __init__(self, personal_dim=4, vascular_dim=6, output_dim=10):
        super().__init__()
        
        # 输入是personal_dim+vascular_dim，输出是output_dim
        self.encoder = nn.Sequential(
            nn.Linear(personal_dim + vascular_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, personal_info, vascular):
        # 连接个人信息和血管属性
        combined = torch.cat([personal_info, vascular], dim=1)  # [B, personal_dim+vascular_dim]
        return self.encoder(combined)  # [B, output_dim]

# 1.8 分支表示编码器 (更复杂的版本，为SBP和DBP创建单独的表示)
class BranchRepresentationEncoder(nn.Module):
    def __init__(self, input_dim, branch_dim=16, hidden_dim=32):
        super().__init__()
        
        # SBP分支 - 更复杂的结构
        self.sbp_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, branch_dim),
            nn.LayerNorm(branch_dim)
        )
        
        # DBP分支 - 更复杂的结构
        self.dbp_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, branch_dim),
            nn.LayerNorm(branch_dim)
        )
    
    def forward(self, features):
        # 为SBP和DBP创建单独的表示
        sbp_representation = self.sbp_branch(features)
        dbp_representation = self.dbp_branch(features)
        
        # 归一化表示（用于对比学习）
        sbp_representation_norm = F.normalize(sbp_representation, p=2, dim=1)
        dbp_representation_norm = F.normalize(dbp_representation, p=2, dim=1)
        
        return sbp_representation_norm, dbp_representation_norm

# 1.9 BP解码器 (血压预测模块)
class BPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=32):
        super().__init__()
        
        # SBP解码器
        self.sbp_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # DBP解码器
        self.dbp_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, sbp_representation, dbp_representation):
        sbp = self.sbp_decoder(sbp_representation).squeeze(-1)
        dbp = self.dbp_decoder(dbp_representation).squeeze(-1)
        
        return torch.stack([sbp, dbp], dim=1)  # [B, 2]

##############################################
# 2. 完整血压对比表示学习模型 (简化版)
##############################################
class IntegratedBPContrastiveModel(nn.Module):
    def __init__(self, in_channels=1, ppg_latent_dim=24, personal_dim=4, vascular_dim=6, 
                 metadata_dim=10, branch_dim=16):
        super().__init__()
        
        # PPG信号编码器
        self.ppg_encoder = PPGEncoder(
            in_channels=in_channels, 
            latent_dim=ppg_latent_dim
        )
        
        # 个人信息和血管属性编码器 (简化版)
        self.personal_vascular_encoder = PersonalVascularEncoder(
            personal_dim=personal_dim,
            vascular_dim=vascular_dim,
            output_dim=metadata_dim
        )
        
        # 分支表示编码器 - 直接从合并的特征创建SBP和DBP表示
        self.branch_encoder = BranchRepresentationEncoder(
            input_dim=ppg_latent_dim + metadata_dim,
            branch_dim=branch_dim,
            hidden_dim=16
        )
        
        # 血压解码器
        self.bp_decoder = BPDecoder(branch_dim)
    
    def forward(self, ppg, personal_info, vascular):
        # 提取PPG特征
        ppg_features = self.ppg_encoder(ppg)  # [B, ppg_latent_dim]
        
        # 提取个人信息和血管属性特征
        metadata_features = self.personal_vascular_encoder(personal_info, vascular)  # [B, metadata_dim]
        
        # 合并特征
        combined_features = torch.cat([ppg_features, metadata_features], dim=1)  # [B, ppg_latent_dim+metadata_dim]
        
        # 生成SBP和DBP专用表示
        sbp_representation, dbp_representation = self.branch_encoder(combined_features)  # [B, branch_dim] x2
        
        # 预测血压
        bp_prediction = self.bp_decoder(sbp_representation, dbp_representation)  # [B, 2] (SBP, DBP)
        
        return {
            'ppg_features': ppg_features,
            'metadata_features': metadata_features,
            'combined_features': combined_features,
            'sbp_representation': sbp_representation,
            'dbp_representation': dbp_representation,
            'bp_prediction': bp_prediction
        }
# 預訓練函數
def pretrain_ppg_model(model, train_loader, optimizer, temporal_criterion, contrastive_criterion, 
                      device, num_epochs=100):
    model.train()
    history = {'loss': [], 'temp_loss': [], 'contr_loss': []}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_temp_loss = 0.0
        running_contr_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 獲取數據
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)
            temporal = batch['temporal'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向傳播
            _, anchor_proj = model(anchor)
            _, positive_proj = model(positive)
            _, negative_proj = model(negative)
            _, temporal_proj = model(temporal)
            
            # 計算損失
            temp_loss = temporal_criterion(anchor_proj, temporal_proj)
            contr_loss = contrastive_criterion(anchor_proj, positive_proj, negative_proj)
            loss = temp_loss + contr_loss
            
            # 反向傳播和優化
            loss.backward()
            optimizer.step()
            
            # 更新運行損失
            running_loss += loss.item()
            running_temp_loss += temp_loss.item()
            running_contr_loss += contr_loss.item()
        
        # 記錄每個epoch的平均損失
        avg_loss = running_loss / len(train_loader)
        avg_temp_loss = running_temp_loss / len(train_loader)
        avg_contr_loss = running_contr_loss / len(train_loader)
        
        history['loss'].append(avg_loss)
        history['temp_loss'].append(avg_temp_loss)
        history['contr_loss'].append(avg_contr_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, "
              f"Temporal Loss: {avg_temp_loss:.4f}, Contrastive Loss: {avg_contr_loss:.4f}")
    
    return history

# 將預訓練模型遷移到最終血壓預測模型
def transfer_pretrained_weights(pretrained_model, target_model):
    # 從預訓練模型中複製編碼器權重到目標模型
    pretrained_dict = pretrained_model.encoder.state_dict()
    target_dict = target_model.ppg_encoder.state_dict()
    
    # 過濾掉不匹配的權重
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}
    
    # 更新目標模型的權重
    target_dict.update(filtered_dict)
    target_model.ppg_encoder.load_state_dict(target_dict)
    
    return target_model

# 繪製損失曲線
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # 繪製損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Total Loss')
    plt.title('Self-Supervised Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 繪製分解的損失
    plt.subplot(1, 2, 2)
    plt.plot(history['temp_loss'], label='Temporal Loss')
    plt.plot(history['contr_loss'], label='Contrastive Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pretrain_history.png')
    plt.show()


# 主函數：預訓練並保存模型
def main():
    print("開始PPG信號自監督學習預訓練流程")
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 設置隨機種子以確保可重複性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 設置數據目錄和輸出目錄
    data_dir = Path("training_data_VitalDB_quality")
    output_dir = Path("pretrain_results")
    output_dir.mkdir(exist_ok=True)
    
    # 找到所有訓練文件
    training_files = list(data_dir.glob("training_*.h5"))
    print(f"找到 {len(training_files)} 個訓練文件")
    
    # 創建預訓練數據集
    transforms = PPGTransforms(gaussian_noise_std=0.01, scaling_range=(0.95, 1.05))
    
    # 為每個訓練文件創建一個滑動窗口數據集
    train_datasets = []
    for h5_path in training_files:
        try:
            ds = SlidingWindowPPGDataset(
                h5_path=h5_path,
                window_size=1250,
                stride=125,
                transforms=transforms
            )
            train_datasets.append(ds)
            print(f"加載數據集 {h5_path.name}, 窗口樣本數: {len(ds)}")
        except Exception as e:
            print(f"加載 {h5_path.name} 時出錯: {e}")
    
    # 合併所有數據集
    combined_train_dataset = ConcatDataset(train_datasets)
    print(f"合併後的預訓練數據集總窗口樣本數: {len(combined_train_dataset)}")
    
    # 創建數據加載器
    train_loader = DataLoader(
        combined_train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 初始化基礎編碼器
    base_encoder = PPGEncoder(in_channels=1, latent_dim=128)
    
    # 創建預訓練模型
    pretrain_model = PPGPretrainModel(base_encoder, projection_dim=128).to(device)
    
    # 打印模型結構和參數數量
    total_params = sum(p.numel() for p in pretrain_model.parameters())
    print(f"預訓練模型總參數數量: {total_params:,}")
    
    # 定義損失和優化器
    temporal_criterion = TemporalConsistencyLoss(temperature=0.1)
    contrastive_criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
    
    # 執行預訓練
    print("\n開始預訓練:")
    history = pretrain_ppg_model(
        model=pretrain_model,
        train_loader=train_loader,
        optimizer=optimizer,
        temporal_criterion=temporal_criterion,
        contrastive_criterion=contrastive_criterion,
        device=device,
        num_epochs=50
    )
    
    # 繪製訓練歷史
    plot_history(history)
    
    # 保存預訓練模型
    pretrain_model_path = output_dir / "pretrained_ppg_model.pth"
    torch.save(pretrain_model.state_dict(), pretrain_model_path)
    print(f"預訓練模型已保存至 {pretrain_model_path}")
    
    # 創建最終的BP預測模型
    bp_model = IntegratedBPContrastiveModel(
        in_channels=1,
        ppg_latent_dim=128,
        personal_dim=4,
        vascular_dim=6,
        metadata_dim=10,
        branch_dim=16
    ).to(device)
    
    # 從預訓練模型遷移權重
    bp_model = transfer_pretrained_weights(pretrain_model, bp_model)
    
    # 保存遷移後的模型
    bp_model_path = output_dir / "pretrained_bp_model.pth"
    torch.save(bp_model.state_dict(), bp_model_path)
    print(f"遷移權重後的血壓模型已保存至 {bp_model_path}")
    
    print("預訓練流程完成!")
    
    return bp_model


if __name__ == "__main__":
    main()