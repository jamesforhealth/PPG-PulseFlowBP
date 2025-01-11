import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import scipy
import sys

import time
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class WindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))
        coords = torch.arange(window_size)
        coords_table = torch.stack(torch.meshgrid([coords])).flatten(1)
        relative_coords = coords_table[:, :, None] - coords_table[:, None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords.squeeze())

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock1D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        H = x.shape[1]
        x = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        # Partition windows
        x_windows = shifted_x.view(-1, self.window_size, self.dim)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        shifted_x = attn_windows.view(-1, H, self.dim)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformer1D(nn.Module):
    def __init__(self, input_size=1250, patch_size=25, dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.patch_embed = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList()
        
        for i in range(len(depths)):
            self.layers.append(SwinTransformerBlock1D(dim=dim*(2**i), depth=depths[i], num_heads=num_heads[i]))
        
        self.norm = nn.LayerNorm(dim * (2**(len(depths)-1)))
        self.head = nn.Linear(dim * (2**(len(depths)-1)), input_size)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.head(x.mean(dim=2))
        return x.unsqueeze(1)  # 恢復通道維度

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer1D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    dataset = AccelToBPDataset(accel_data, bp_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train(model, dataloader, criterion, optimizer, device, scaler)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()