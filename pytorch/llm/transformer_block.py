import math
import numpy as np

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=80):
        '''
        PE(pos, 2i) = sin(pos/10000^(2i/embedding_dim))
        PE(pos, 2i+1) = cos(pos/10000^(2i/embedding_dim))
        '''
        super(PositionalEncoding, self).__init__()
        self.dimension = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                exp = 2 * i / embedding_dim
                exp_val = 10000 ** exp
                pe[pos, i] = math.sin(pos / exp_val)
                pe[pos, i+1] = math.cos(pos / exp_val)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x1 = x * math.sqrt(self.dimension)
        seq_len = x.size(1)
        x1 = x1 + self.pe[:, :seq_len]
        return x1


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dimension = embedding_dim
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        kt = k.transpose(2,1)
        score = torch.bmm(q, kt) / math.sqrt(self.dimension)
        score = self.dropout(score)
        attention_score = torch.softmax(score, dim=1)
        y = torch.bmm(attention_score, v)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.dimension = embedding_dim
        self.wq_list = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(heads)])
        self.wk_list = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(heads)])
        self.wv_list = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(heads)])
        self.scale = self.dimension ** 0.5
        self.wo = nn.Linear(self.embedding_dim * heads, embedding_dim)

    def forward(self, x):
        pass


if __name__ == '__main__':
    pe = PositionalEncoding(128, 80)
    pe = pe.cuda()
    x = torch.randn([2, 12, 128]).cuda()
    y = pe(x)
    print(y.shape)
