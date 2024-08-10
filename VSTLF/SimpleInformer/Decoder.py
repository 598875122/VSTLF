import torch.nn as nn
from attn import ProbAttention
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, dim_in, num_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attention = ProbAttention(hidden_dim, hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)
        self.enc_attention = ProbAttention(hidden_dim, hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x, enc_out):
        # Self attention
        self_att = self.dropout1(self.self_attention(x))
        x = self.norm1(x + self_att)

        # Encoder-Decoder attention
        enc_att = self.dropout2(self.enc_attention(x, k=enc_out, v=enc_out))  # Pass encoder outputs as keys and values
        y = x = self.norm2(x + enc_att)
        y = self.dropout3(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout3(self.conv2(y).transpose(-1, 1))

        x = self.norm3(x + y)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, dim_in, num_heads, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim_in, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


