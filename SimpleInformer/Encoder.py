import torch
import torch.nn as nn
from attn import ProbAttention
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, dim_in, num_heads, hidden_dim, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)
        self.attention = ProbAttention(hidden_dim, hidden_dim, num_heads, dropout)  # 使用ProbAttention而不是Attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = F.relu

    def forward(self, x):
        # print("attn",x.shape)
        att = self.dropout(self.attention(x))
        # print(x.shape)
        x = x + att
        # ff = self.dropout2(self.feedforward(x))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, num_layers, dim_in, num_heads, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim_in, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
