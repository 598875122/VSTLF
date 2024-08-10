import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from Embed import TimeFeatureEmbedding, DataEmbedding,DataEmbedding_2
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, e_layers, d_layers, hidden_dim, n_heads,
                 dropout):
        super(Informer, self).__init__()
        self.data_embedding = DataEmbedding(c_in=enc_in, d_model=hidden_dim)
        self.data_embedding_2 = DataEmbedding_2(c_in=enc_in, d_model=hidden_dim)
        self.encoder = Encoder(e_layers, hidden_dim, n_heads, hidden_dim, dropout)
        self.decoder = Decoder(d_layers, hidden_dim, n_heads, hidden_dim, dropout)
        self.projection = nn.Linear(hidden_dim, 1)
        # self.projection1 = nn.Linear(128, 1)
        # self.gelu = nn.GELU()

    def forward(self, src_seq, tgt_seq, time_in, enc_self_mask=None):
        enc_input = self.data_embedding(src_seq, time_in)
        dec_input = self.data_embedding(tgt_seq, time_in)
        enc_out = self.encoder(enc_input)
        dec_out = self.decoder(dec_input, enc_out)
        output = self.projection(dec_out)
        # output = self.gelu(output)
        # output = self.projection1(output)

        return output[:, -1, :]