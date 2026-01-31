from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameCNN(nn.Module):
    def __init__(self, in_ch=3, ch1=64, ch2=128, emb_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B,3,H,W)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = self.proj(x)
        return x  # (B, emb_dim)

class BranchEncoder(nn.Module):
    def __init__(self, cnn_channels, emb_dim, bilstm_hidden, gru_hidden, fusion_lstm_hidden, dropout):
        super().__init__()
        ch1, ch2 = cnn_channels
        self.cnn = FrameCNN(ch1=ch1, ch2=ch2, emb_dim=emb_dim, dropout=dropout)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=bilstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.time_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=2*bilstm_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.fusion_lstm = nn.LSTM(
            input_size=gru_hidden,
            hidden_size=fusion_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.out_dim = fusion_lstm_hidden

    def forward(self, clip):
        # clip: (B,T,3,H,W)
        B, T, C, H, W = clip.shape
        x = clip.reshape(B*T, C, H, W)
        f = self.cnn(x)               # (B*T, emb)
        f = f.reshape(B, T, -1)       # (B,T,emb)
        h, _ = self.bilstm(f)         # (B,T,2*h)
        # temporal max-pool along time using MaxPool1d expects (B, C, T)
        h_t = h.transpose(1, 2)       # (B,2*h,T)
        h_t = self.time_pool(h_t)     # (B,2*h,T/2)
        h = h_t.transpose(1, 2)       # (B,T/2,2*h)
        h = self.dropout(h)
        g, _ = self.gru(h)            # (B,T/2,gru_hidden)
        # use last hidden state after a small fusion lstm for compactness
        u, (hn, cn) = self.fusion_lstm(g)  # hn: (1,B,hidden)
        feat = hn[-1]                      # (B, fusion_hidden)
        feat = self.dropout(feat)
        return feat

class TinyHybridAttentionModel(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, branch_cfgs: List[Dict]):
        super().__init__()
        self.branches = nn.ModuleList([
            BranchEncoder(
                cnn_channels=cfg['cnn_channels'],
                emb_dim=emb_dim,
                bilstm_hidden=cfg['bilstm_hidden'],
                gru_hidden=cfg['gru_hidden'],
                fusion_lstm_hidden=cfg['fusion_lstm_hidden'],
                dropout=cfg['dropout']
            )
            for cfg in branch_cfgs
        ])
        fused_dim = sum(b.out_dim for b in self.branches)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fused_dim//2, num_classes)
        )

    def forward(self, clip):
        feats = [b(clip) for b in self.branches]
        fused = torch.cat(feats, dim=1)
        logits = self.head(fused)
        return logits
