import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

##############################################
# 0. 数据集定义
##############################################
class PPGBPDataset(Dataset):
    """读取h5文件中的PPG信号和血压值等数据"""
    def __init__(self, h5_path, include_personal_info=True, include_vascular=True):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.include_personal_info = include_personal_info
        self.include_vascular = include_vascular
        
        with h5py.File(self.h5_path, 'r') as f:
            # 必须存在的数据
            self.ppg = torch.from_numpy(f['ppg'][:]).float()  # (N,1250)
            
            # 血压标签
            if 'segsbp' in f and 'segdbp' in f:
                self.segsbp = torch.from_numpy(f['segsbp'][:]).float()  # (N,)
                self.segdbp = torch.from_numpy(f['segdbp'][:]).float()  # (N,)
                self.bp_label = torch.stack([self.segsbp, self.segdbp], dim=1)  # (N,2)
            else:
                raise ValueError("数据集必须包含segsbp和segdbp标签！")
            
            # 加载血管属性 - 6个特征
            if self.include_vascular and 'vascular_properties' in f:
                self.vascular = torch.from_numpy(f['vascular_properties'][:]).float()
                
                # 确保有6个特征
                if self.vascular.shape[1] < 6:
                    padded = torch.zeros((len(self.ppg), 6))
                    padded[:, :self.vascular.shape[1]] = self.vascular
                    self.vascular = padded
            else:
                self.vascular = torch.zeros((len(self.ppg), 6))  # 默认6个特征
            
            # 加载个人信息 - 4个特征
            if self.include_personal_info and 'personal_info' in f:
                self.personal_info = torch.from_numpy(f['personal_info'][:]).float()
                
                # 确保有4个特征
                if self.personal_info.shape[1] < 4:
                    padded = torch.zeros((len(self.ppg), 4))
                    padded[:, :self.personal_info.shape[1]] = self.personal_info
                    self.personal_info = padded
            else:
                self.personal_info = torch.zeros((len(self.ppg), 4))  # 默认4个特征
        
        self.N = len(self.ppg)
        
        # 添加channel维度
        self.ppg = self.ppg.unsqueeze(1)  # (N,1,1250)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx],              # (1,1250)
            'bp_label': self.bp_label[idx],    # (2,)
            'vascular': self.vascular[idx],    # (6,)
            'personal_info': self.personal_info[idx]  # (4,)
        }

##############################################
# 1. 模型组件
##############################################
# 1.1 基本1D卷积块
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

# 1.6 PPG编码器
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

##############################################
# 3. 损失函数
##############################################
class BPContrastiveLoss(nn.Module):
    """
    改进版血压对比损失：确保规模与回归损失相匹配
    """
    def __init__(self, temperature=0.1, sigma=5.0, bp_type='sbp', scale_factor=1e-6):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.bp_type = bp_type
        self.scale_factor = scale_factor  # 添加缩放因子控制损失规模

    def forward(self, representations, bp_labels):
        """
        representations: [B, rep_dim]，已经归一化
        bp_labels: [B, 2]，取其中一列作为血压值
        """
        batch_size = representations.size(0)
        
        # 选择血压值（例如 SBP 或 DBP）
        if self.bp_type == 'sbp':
            bp_values = bp_labels[:, 0].unsqueeze(1)  # [B, 1]
        else:
            bp_values = bp_labels[:, 1].unsqueeze(1)  # [B, 1]
        
        # 计算目标距离矩阵 (L2距离)
        bp_dist = torch.cdist(bp_values, bp_values, p=2)  # [B, B]
        # 高斯核转换为相似度
        target_sim = torch.exp(-bp_dist / (2 * self.sigma**2))
        # 对每个样本归一化
        target_sim = target_sim / target_sim.sum(dim=1, keepdim=True)
        
        # 计算表示向量的余弦相似度矩阵
        logits = torch.mm(representations, representations.t()) / self.temperature  # [B, B]
        # 为避免自匹配，将对角线设为 -inf
        mask = torch.eye(batch_size, device=logits.device).bool()
        logits = logits.masked_fill(mask, -1e9)
        
        # 使用 log_softmax 计算对数概率分布
        pred_log_prob = F.log_softmax(logits, dim=1)  # [B, B]
        
        # 使用 KL 散度计算损失并应用缩放因子
        loss = F.kl_div(pred_log_prob, target_sim, reduction='batchmean') * self.scale_factor
        
        return loss

class PhysicalConsistencyLoss(nn.Module):
    """物理一致性约束损失"""
    def __init__(self, weight_pulse=1.0, weight_range=0.5):
        super().__init__()
        self.weight_pulse = weight_pulse  # 脉压约束权重
        self.weight_range = weight_range  # 范围约束权重
    
    def forward(self, bp_prediction):
        # 从预测中提取SBP和DBP
        sbp = bp_prediction[:, 0]
        dbp = bp_prediction[:, 1]
        
        # 1. SBP必须大于DBP（脉压约束）- 脉压应≥5 mmHg
        pulse_pressure_loss = F.relu(-(sbp - dbp) + 5).mean() * self.weight_pulse
        
        # 2. SBP和DBP应在合理的生理范围内
        sbp_range_loss = (F.relu(70 - sbp) + F.relu(sbp - 200)).mean() * self.weight_range
        dbp_range_loss = (F.relu(40 - dbp) + F.relu(dbp - 120)).mean() * self.weight_range
        
        # 组合损失
        total_loss = pulse_pressure_loss + sbp_range_loss + dbp_range_loss
        
        return total_loss

class BPLoss(nn.Module):
    """综合血压预测损失"""
    def __init__(self, contrastive_weight=0.1, physical_weight=0.2):
        super().__init__()
        # 使用L1Loss代替SmoothL1Loss
        self.regression_loss = nn.L1Loss()
        # 对比损失，添加比例缩放因子
        self.sbp_contrastive_loss = BPContrastiveLoss(bp_type='sbp', scale_factor=1e-6)
        self.dbp_contrastive_loss = BPContrastiveLoss(bp_type='dbp', scale_factor=1e-6)
        self.physical_loss = PhysicalConsistencyLoss()
        
        # 调整权重，确保回归损失占主导
        self.contrastive_weight = contrastive_weight
        self.physical_weight = physical_weight
    
    def forward(self, outputs, bp_labels):
        # 回归损失
        reg_loss = self.regression_loss(outputs['bp_prediction'], bp_labels)
        
        # 对比损失
        sbp_contr_loss = self.sbp_contrastive_loss(outputs['sbp_representation'], bp_labels)
        dbp_contr_loss = self.dbp_contrastive_loss(outputs['dbp_representation'], bp_labels)
        contr_loss = (sbp_contr_loss + dbp_contr_loss) / 2.0
        
        # 物理一致性损失
        phys_loss = self.physical_loss(outputs['bp_prediction'])
        
        # 总损失 - 确保回归损失占主导
        total_loss = reg_loss + self.contrastive_weight * contr_loss + self.physical_weight * phys_loss
        
        return total_loss, reg_loss, contr_loss, phys_loss

##############################################
# 4. 训练和验证函数
##############################################
def train_epoch(model, data_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    
    running_loss = 0.0
    running_reg_loss = 0.0
    running_contr_loss = 0.0
    running_phys_loss = 0.0
    sbp_errors = []
    dbp_errors = []
    
    pbar = tqdm(data_loader, desc="训练")
    for batch in pbar:
        # 获取数据
        ppg = batch['ppg'].to(device)
        bp_label = batch['bp_label'].to(device)
        personal_info = batch['personal_info'].to(device)
        vascular = batch['vascular'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(ppg, personal_info, vascular)
        
        # 计算损失
        loss, reg_loss, contr_loss, phys_loss = criterion(outputs, bp_label)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 更新运行损失
        running_loss += loss.item()
        running_reg_loss += reg_loss.item()
        running_contr_loss += contr_loss.item()
        running_phys_loss += phys_loss.item()
        
        # 计算SBP和DBP误差
        pred_bp = outputs['bp_prediction'].detach()
        sbp_error = torch.abs(pred_bp[:, 0] - bp_label[:, 0])
        dbp_error = torch.abs(pred_bp[:, 1] - bp_label[:, 1])
        
        sbp_errors.extend(sbp_error.cpu().numpy())
        dbp_errors.extend(dbp_error.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'sbp_mae': sbp_error.mean().item(),
            'dbp_mae': dbp_error.mean().item()
        })
    
    # 计算平均损失和MAE
    avg_loss = running_loss / len(data_loader)
    avg_reg_loss = running_reg_loss / len(data_loader)
    avg_contr_loss = running_contr_loss / len(data_loader)
    avg_phys_loss = running_phys_loss / len(data_loader)
    sbp_mae = np.mean(sbp_errors)
    dbp_mae = np.mean(dbp_errors)
    
    return avg_loss, avg_reg_loss, avg_contr_loss, avg_phys_loss, sbp_mae, dbp_mae

def validate(model, data_loader, criterion, device):
    """验证模型"""
    model.eval()
    
    running_loss = 0.0
    running_reg_loss = 0.0
    running_contr_loss = 0.0
    running_phys_loss = 0.0
    sbp_errors = []
    dbp_errors = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="验证"):
            # 获取数据
            ppg = batch['ppg'].to(device)
            bp_label = batch['bp_label'].to(device)
            personal_info = batch['personal_info'].to(device)
            vascular = batch['vascular'].to(device)
            
            # 前向传播
            outputs = model(ppg, personal_info, vascular)
            
            # 计算损失
            loss, reg_loss, contr_loss, phys_loss = criterion(outputs, bp_label)
            
            # 更新运行损失
            running_loss += loss.item()
            running_reg_loss += reg_loss.item()
            running_contr_loss += contr_loss.item()
            running_phys_loss += phys_loss.item()
            
            # 计算SBP和DBP误差
            pred_bp = outputs['bp_prediction']
            sbp_error = torch.abs(pred_bp[:, 0] - bp_label[:, 0])
            dbp_error = torch.abs(pred_bp[:, 1] - bp_label[:, 1])
            
            sbp_errors.extend(sbp_error.cpu().numpy())
            dbp_errors.extend(dbp_error.cpu().numpy())
    
    # 计算平均损失和MAE
    avg_loss = running_loss / len(data_loader)
    avg_reg_loss = running_reg_loss / len(data_loader)
    avg_contr_loss = running_contr_loss / len(data_loader)
    avg_phys_loss = running_phys_loss / len(data_loader)
    sbp_mae = np.mean(sbp_errors)
    dbp_mae = np.mean(dbp_errors)
    
    return avg_loss, avg_reg_loss, avg_contr_loss, avg_phys_loss, sbp_mae, dbp_mae

##############################################
# 5. 主训练函数
##############################################
def train_model(train_loader, val_loader, model, criterion, device, 
                learning_rate=1e-3, num_epochs=100, patience=15, 
                model_save_path='best_bp_model.pth'):
    """
    训练血压预测模型，同时记录各项损失变化和当前学习率。
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, min_lr=1e-6, verbose=True
    )
    
    history = {
        'lr': [],
        'train_loss': [], 'train_reg_loss': [], 'train_contr_loss': [], 'train_phys_loss': [],
        'val_loss': [], 'val_reg_loss': [], 'val_contr_loss': [], 'val_phys_loss': [],
        'train_sbp_mae': [], 'train_dbp_mae': [], 'val_sbp_mae': [], 'val_dbp_mae': []
    }
    
    best_val_loss = float('inf')
    best_bp_mae = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"当前学习率: {current_lr:.6f}")
        
        train_loss, train_reg_loss, train_contr_loss, train_phys_loss, train_sbp_mae, train_dbp_mae = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_reg_loss, val_contr_loss, val_phys_loss, val_sbp_mae, val_dbp_mae = validate(
            model, val_loader, criterion, device
        )
        curr_bp_mae = (val_sbp_mae + val_dbp_mae) / 2
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_reg_loss'].append(train_reg_loss)
        history['train_contr_loss'].append(train_contr_loss)
        history['train_phys_loss'].append(train_phys_loss)
        history['train_sbp_mae'].append(train_sbp_mae)
        history['train_dbp_mae'].append(train_dbp_mae)
        
        history['val_loss'].append(val_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['val_contr_loss'].append(val_contr_loss)
        history['val_phys_loss'].append(val_phys_loss)
        history['val_sbp_mae'].append(val_sbp_mae)
        history['val_dbp_mae'].append(val_dbp_mae)
        
        print(f"训练: 损失={train_loss:.4f}, Reg Loss={train_reg_loss:.4f}, Contr Loss={train_contr_loss:.4f}, Phys Loss={train_phys_loss:.4f}")
        print(f"训练: SBP MAE={train_sbp_mae:.2f}, DBP MAE={train_dbp_mae:.2f}")
        print(f"验证: 损失={val_loss:.4f}, Reg Loss={val_reg_loss:.4f}, Contr Loss={val_contr_loss:.4f}, Phys Loss={val_phys_loss:.4f}")
        print(f"验证: SBP MAE={val_sbp_mae:.2f}, DBP MAE={val_dbp_mae:.2f}, 平均 BP MAE={(val_sbp_mae+val_dbp_mae)/2:.2f}")
        
        if curr_bp_mae < best_bp_mae:
            best_bp_mae = curr_bp_mae
            torch.save(model.state_dict(), model_save_path)
            print(f"保存新的最佳模型, 平均 BP MAE={best_bp_mae:.2f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"无改善轮数: {no_improve_epochs}/{patience}")
        
        if no_improve_epochs >= patience:
            print(f"早停! {patience}轮未改善.")
            break
    
    return history

# 绘图函数保持不变
def plot_history(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    
    # 学习率变化
    plt.subplot(2, 2, 1)
    plt.plot(history['lr'], marker='o')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    
    # 总损失
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('总损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 损失分解
    plt.subplot(2, 2, 3)
    plt.plot(history['train_reg_loss'], label='Train Reg Loss')
    plt.plot(history['val_reg_loss'], label='Val Reg Loss')
    plt.plot(history['train_contr_loss'], label='Train Contr Loss', linestyle='--')
    plt.plot(history['val_contr_loss'], label='Val Contr Loss', linestyle='--')
    plt.plot(history['train_phys_loss'], label='Train Phys Loss', linestyle=':')
    plt.plot(history['val_phys_loss'], label='Val Phys Loss', linestyle=':')
    plt.title('损失分解')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # MAE
    plt.subplot(2, 2, 4)
    plt.plot(history['train_sbp_mae'], label='Train SBP MAE')
    plt.plot(history['val_sbp_mae'], label='Val SBP MAE')
    plt.plot(history['train_dbp_mae'], label='Train DBP MAE')
    plt.plot(history['val_dbp_mae'], label='Val DBP MAE')
    plt.title('血压 MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

##############################################
# 7. 预测函数
##############################################
def predict_bp(ppg_signal, personal_info=None, vascular=None, model_path='best_bp_model.pth', device=None):
    """
    使用训练好的模型进行血压预测
    
    参数:
    - ppg_signal: numpy数组, 形状为 (1250,) 或 (1, 1250)
    - personal_info: 个人信息, numpy数组, 形状为 (4,)，或None
    - vascular: 血管属性, numpy数组, 形状为 (6,)，或None
    - model_path: 模型文件路径
    - device: 计算设备
    
    返回:
    - sbp: 收缩压预测值
    - dbp: 舒张压预测值
    - sbp_rep, dbp_rep: SBP和DBP的表示向量
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = IntegratedBPContrastiveModel(
        in_channels=1, ppg_latent_dim=128, personal_dim=4, 
        vascular_dim=6, metadata_dim=10, branch_dim=16
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 处理输入
    if ppg_signal.ndim == 1:
        ppg_signal = ppg_signal.reshape(1, 1, -1)  # [1, 1, 1250]
    elif ppg_signal.ndim == 2:
        ppg_signal = ppg_signal.reshape(1, ppg_signal.shape[0], ppg_signal.shape[1])
    
    ppg_tensor = torch.tensor(ppg_signal, dtype=torch.float32).to(device)
    
    # 处理个人信息
    if personal_info is None:
        personal_info = np.zeros(4)
    personal_tensor = torch.tensor(personal_info, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 处理血管属性
    if vascular is None:
        vascular = np.zeros(6)
    vascular_tensor = torch.tensor(vascular, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(ppg_tensor, personal_tensor, vascular_tensor)
    
    # 获取结果
    sbp = outputs['bp_prediction'][0, 0].item()
    dbp = outputs['bp_prediction'][0, 1].item()
    sbp_rep = outputs['sbp_representation'][0].cpu().numpy()
    dbp_rep = outputs['dbp_representation'][0].cpu().numpy()
    
    return sbp, dbp, sbp_rep, dbp_rep

##############################################
# 8. 主函数
##############################################
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置数据目录和输出目录
    data_dir = Path("training_data_VitalDB_quality")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 找到所有训练文件
    training_files = list(data_dir.glob("training_*.h5"))
    print(f"找到 {len(training_files)} 个训练文件")
    
    # 创建数据集
    train_datasets = []
    for h5_path in training_files:
        try:
            ds = PPGBPDataset(h5_path, include_personal_info=True, include_vascular=True)
            train_datasets.append(ds)
            print(f"加载数据集 {h5_path.name}, 样本数: {len(ds)}")
        except Exception as e:
            print(f"加载 {h5_path.name} 时出错: {e}")
    
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = PPGBPDataset(data_dir / "validation.h5", include_personal_info=True, include_vascular=True)
    test_dataset = PPGBPDataset(data_dir / "test.h5", include_personal_info=True, include_vascular=True)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 创建模型
    model = IntegratedBPContrastiveModel(
        in_channels=1,
        ppg_latent_dim=28,
        personal_dim=4,
        vascular_dim=6,
        metadata_dim=10,
        branch_dim=12
    ).to(device)
    
    # 打印模型结构和参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")
    
    # 创建损失函数
    criterion = BPLoss(contrastive_weight=0.1, physical_weight=0.02)
    
    # 训练模型
    model_save_path = output_dir / "best_bp_model.pth"
    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        learning_rate=1e-3,
        num_epochs=100,
        patience=30,  # 早停耐心值
        model_save_path=model_save_path
    )
    
    # 绘制训练历史
    plot_history(history)
    
    # 在测试集上评估最佳模型
    best_model = IntegratedBPContrastiveModel(
        in_channels=1,
        ppg_latent_dim=28,
        personal_dim=4,
        vascular_dim=6,
        metadata_dim=10,
        branch_dim=12
    ).to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    print("sum of parameters: ", sum(p.numel() for p in best_model.parameters()))
    # 测试
    print("\n在测试集上评估最佳模型:")
    test_loss, test_reg_loss, test_contr_loss, test_phys_loss, test_sbp_mae, test_dbp_mae = validate(
        best_model, test_loader, criterion, device
    )
    
    print(f"测试结果 - 总损失: {test_loss:.4f}")
    print(f"测试结果 - 回归损失: {test_reg_loss:.4f}")
    print(f"测试结果 - 对比损失: {test_contr_loss:.4f}")
    print(f"测试结果 - 物理约束损失: {test_phys_loss:.4f}")
    print(f"测试结果 - SBP MAE: {test_sbp_mae:.2f} mmHg")
    print(f"测试结果 - DBP MAE: {test_dbp_mae:.2f} mmHg")
    print(f"测试结果 - 平均 BP MAE: {(test_sbp_mae + test_dbp_mae) / 2:.2f} mmHg")
    
    # 保存测试结果
    results = {
        "test_loss": test_loss,
        "test_reg_loss": test_reg_loss,
        "test_contr_loss": test_contr_loss,
        "test_phys_loss": test_phys_loss,
        "test_sbp_mae": test_sbp_mae,
        "test_dbp_mae": test_dbp_mae,
        "test_avg_mae": (test_sbp_mae + test_dbp_mae) / 2
    }
    
    # 保存为JSON
    import json
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"测试结果已保存到 {output_dir / 'test_results.json'}")

if __name__ == "__main__":
    main()
