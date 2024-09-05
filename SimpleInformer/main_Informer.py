import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from Embed import TimeFeatureEmbedding, DataEmbedding, DataEmbedding_2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, e_layers, d_layers, hidden_dim, n_heads,
                 dropout):
        super(Informer, self).__init__()
        # Initialize the data embeddings for input and output sequences
        self.data_embedding = DataEmbedding(c_in=enc_in, d_model=hidden_dim)
        self.data_embedding_2 = DataEmbedding_2(c_in=enc_in, d_model=hidden_dim)
        # Initialize the encoder and decoder
        self.encoder = Encoder(e_layers, hidden_dim, n_heads, hidden_dim, dropout)
        self.decoder = Decoder(d_layers, hidden_dim, n_heads, hidden_dim, dropout)
        # Initialize the projection layer to map decoder output to the final output
        self.projection = nn.Linear(hidden_dim, 1)
        # Uncomment these if additional transformations are needed
        # self.projection1 = nn.Linear(128, 1)
        # self.gelu = nn.GELU()

    def forward(self, src_seq, tgt_seq, time_in, enc_self_mask=None):
        # Embed the input and target sequences
        enc_input = self.data_embedding(src_seq, time_in)
        dec_input = self.data_embedding(tgt_seq, time_in)
        # Pass the embedded input through the encoder
        enc_out = self.encoder(enc_input)
        # Pass the embedded target through the decoder, using encoder output
        dec_out = self.decoder(dec_input, enc_out)
        # Project the decoder output to the final output space
        output = self.projection(dec_out)
        # Uncomment these lines if additional transformations are needed
        # output = self.gelu(output)
        # output = self.projection1(output)

        # Return the output for the last time step
        return output[:, -1, :]
