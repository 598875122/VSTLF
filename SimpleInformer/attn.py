import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularCausalMask(nn.Module):
    def forward(self, tensor):
        shape = tensor.size()
        if len(shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (batch_size, num_heads, seq_length, seq_length)")

        # Get sequence length
        seq_length = shape[2]

        # Create a lower triangular matrix as the mask
        mask = torch.tril(torch.ones(seq_length, seq_length), diagonal=0)

        # Expand the mask to include batch size and number of heads dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)

        # Expand the mask to match the batch size and number of heads of the input tensor
        mask = mask.expand(shape[0], shape[1], seq_length, seq_length)  # (batch_size, num_heads, seq_length, seq_length)

        if tensor.is_cuda:
            mask = mask.cuda()

        return mask


class ProbMask(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, tensor):
        B, H, L, _ = tensor.size()
        ones = torch.ones(L, L)
        a_mask = ones.tril(-1)[None, None, :, :].expand(B, H, L, L)
        p_mask = torch.bernoulli(a_mask * self.dropout)
        return p_mask.to(tensor.device)


class ProbAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout=0.0):
        super().__init__()
        self.head_dim = dim_out // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_out)
        self.linear_k = nn.Linear(dim_in, dim_out)
        self.linear_v = nn.Linear(dim_in, dim_out)
        self.linear_o = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = TriangularCausalMask()
        self.prob_mask = ProbMask(dropout)

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = q
        Q = self.linear_q(q)
        K = self.linear_k(k)
        V = self.linear_v(v)
        Q = Q.view(Q.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(K.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(V.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        # Apply causal mask
        causal_mask = self.causal_mask(energy)
        energy = energy.masked_fill(causal_mask == 0, float('-inf'))
        attention_weights = torch.softmax(energy, dim=-1)
        # Apply dropout with probability mask
        prob_mask = self.prob_mask(attention_weights)
        attention_weights = attention_weights.masked_fill(prob_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_weights * prob_mask, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(Q.size(0), -1, self.num_heads * self.head_dim)
        return self.linear_o(out)

# Input-output dimension explanation:
# Input:
#   q: Query tensor of shape (batch_size, seq_len_q, d_model)
#   k: Key tensor of shape (batch_size, seq_len_k, d_model)
#   v: Value tensor of shape (batch_size, seq_len_v, d_model)
#   attn_mask (optional): Attention mask of shape (batch_size, num_heads, seq_len_q, seq_len_k)
# Output:
#   output: Output tensor of shape (batch_size, seq_len_q, d_model)


# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, dim)

#     def forward(self, x):
#         return self.linear2(torch.relu(self.linear1(x)))
