import torch
import torch.nn as nn

import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    # heads = 12, d_model = 768
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.dk = self.d_model // heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, q, k, v, mask=None):
        # q [B, H, S, D]
        score = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            score.masked_fill_(mask==0, -1e10)
        attn_score = F.softmax(score, dim=-1)
        attn_score = self.dropout(attn_score)
        out = torch.matmul(attn_score, v)
        return out

    def forward(self, q, k, v, mask=None):
        # q -> [B, S, D] q_proj -> [B, H, S, dk]
        q_proj = self.q_linear(q).view(q.size(0), -1, self.heads, self.dk).transpose(1,2)
        k_proj = self.q_linear(q).view(k.size(0), -1, self.heads, self.dk).transpose(1,2)
        v_proj = self.q_linear(q).view(v.size(0), -1, self.heads, self.dk).transpose(1,2)
        # [B, H, S, D]
        attention_out = self.attention(q_proj, k_proj, v_proj, mask)
        x = attention_out.transpose(1,2).contiguous().view(q.size(0), q.size(1), -1)
        return self.out(x)


class MultiQueryAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiQueryAttention, self).__init__()
        self.dk = d_model // heads
        self.heads = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.dk)
        self.v_linear = nn.Linear(d_model, self.dk)

        self.out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, q, k, v, mask=None):
        # q -> [B,H,S,D] k -> [B,1,S,D]
        score = torch.matmul(q, k.transpose(-1,-2))
        if mask is not None:
            score.masked_fill_(mask == 0, -1e10)
        attn_score = F.softmax(score, dim=-1)
        out = torch.matmul(attn_score, v)
        return out

    def forward(self, q, k, v, mask=None):
        # q, k, v -> [B, S, D], q_proj -> [B, H, S, dk]
        batch_size = q.size(0)
        q_proj = self.q_linear(q).view(batch_size, -1, self.heads, self.dk).transpose(1, 2)
        # k_proj -> [B, 1, S, D]
        k_proj = self.k_linear(k).view(batch_size, -1, 1, self.dk).transpose(1, 2)
        v_proj = self.v_linear(v).view(batch_size, -1,1 , self.dk).transpose(1, 2)
        attn_out = self.attention(q_proj, k_proj, v_proj, mask)
        # attn_out -> [B, H, S, D]
        x = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, q.size(-1))
        return self.out(x)


class GroupQueryAttention(nn.Module):
    def __init__(self, groups=2, heads=12, d_model=768, dropout=0.1):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.groups = groups
        self.dk = d_model // heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.dk * self.groups)
        self.v_linear = nn.Linear(d_model, self.dk * self.groups)

        self.out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, q, k, v, mask=None):
        # q -> [B, H, S, D], k -> [B, H, S, D]
        score = torch.matmul(q, k.transpose(-2,-1))
        if mask is not None:
            score.masked_fill_(mask==0, -1e10)
        attn_score = F.softmax(score, dim=-1)
        attn_score = self.dropout(attn_score)
        out = torch.matmul(attn_score, v)
        return out

    def forward(self, q, k, v, mask=None):
        # q -> [B, S, D], q_proj -> [B, H, S, dk]
        bs = q.size(0)
        q_proj = self.q_linear(q).view(bs, -1, self.heads, self.dk).transpose(1, 2)
        # k_proj -> [B, G, S, dk]
        k_proj = self.k_linear(k).view(bs, -1, self.groups, self.dk).repeat(1,1,self.heads//self.groups, 1).transpose(1, 2)
        v_proj = self.v_linear(v).view(bs, -1, self.groups, self.dk).repeat(1,1,self.heads//self.groups, 1).transpose(1, 2)
        attn_out = self.attention(q_proj, k_proj, v_proj, mask)
        x = attn_out.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        return self.out(x)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        # p(i, 2t) = sin(i/ (10000 ^ (2t/d))), p(i, 2t) = cos(i/ (10000 ^ (2t/d)))
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pe = torch.zeros(seq_len, d_model)

        pos_vec = np.array([self.get_pos_vec(i) for i in range(seq_len)])
        pos_vec = torch.from_numpy(pos_vec)

        self.pe[:, 0::2] = torch.sin(pos_vec[:, 0::2])
        self.pe[:, 1::2] = torch.cos(pos_vec[:, 1::2])

        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)

    def get_pos_vec(self, pos):
        return [pos / np.power(10000, 2*(t//2) / self.d_model) for t in range(self.d_model)]

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, seq_len, base=10000.0):
        super(RotaryPositionEmbedding, self).__init__()
        freqs = 1.0 / (torch.tensor(base) ** (torch.arange(0, d_model, 2) / d_model))
        seqs = torch.arange(0, seq_len)
        freqs = torch.outer(seqs, freqs)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.freqs_complex = freqs_complex
        self.register_buffer('freqs', freqs_complex)

    def forward(self, x):
        # B, S, D
        x1 = x.reshape(*x.shape[:-1], -1, 2)
        x2 = torch.view_as_complex(x1)
        y = self.freqs_complex * x2
        y = torch.view_as_real(y).flatten(2)
        return y



if __name__ == '__main__':
    model = GroupQueryAttention(4, 12,768)
    q = torch.randn(2, 128, 768)
    k = torch.randn(2, 128, 768)
    v = torch.randn(2, 128, 768)
    y = model(q, k, v)
    print(y.shape)
