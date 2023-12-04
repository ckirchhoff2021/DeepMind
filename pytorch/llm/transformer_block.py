import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=80):
        '''
        PE(pos, 2i) = sin(pos/10000^(2i/embedding_dim))
        PE(pos, 2i+1) = cos(pos/10000^(2i/embedding_dim))
        '''
        super(PositionalEncoder, self).__init__()
        self.dimension = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                exp = 2 * i / embedding_dim
                exp_val = 10000 ** exp
                pe[pos, i] = math.sin(pos / exp_val)
                pe[pos, i+1] = math.cos(pos / exp_val)
        pe = pe.unsqueeze(0)
        # register buffer, which wouldn't update during training steps
        self.register_buffer('pe', pe)

    def forward(self, x):
        # scale the word embedding, of which shape is (batch_size, seq_len, hidden_dim)
        x = x * math.sqrt(self.dimension)
        seq_len = x.size(1)
        x1 = x + self.pe[:, :seq_len]
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
        self.hidden_dim = embedding_dim

        self.wq = nn.Linear(self.hidden_dim, self.hidden_dim * self.heads)
        self.wk = nn.Linear(self.hidden_dim, self.hidden_dim * self.heads)
        self.wv = nn.Linear(self.hidden_dim, self.hidden_dim * self.heads)

        self.scale = self.hidden_dim ** 0.5
        self.out = nn.Linear(self.hidden_dim * self.heads, self.hidden_dim)

    @staticmethod
    def attention(q, k, v, hidden_dim, heads, mask=None, dropout=None):
        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(hidden_dim)

        if mask is not None:
            mask_lists = [mask] * heads
            mask = torch.cat(mask_lists, dim=1)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        output = torch.bmm(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.wk(q).view(bs, -1, self.hidden_dim)
        k = self.wq(k).view(bs, -1, self.hidden_dim)
        v = self.wv(v).view(bs, -1, self.hidden_dim)

        scores = self.attention(q, k, v, self.hidden_dim, self.heads, mask, self.dropout)
        scores = scores.view(bs, -1, self.hidden_dim * self.heads)
        output = self.out(scores)
        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, middle_dim=2048 ,dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, middle_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(middle_dim, hidden_dim)

    def forward(self, x):
        y = self.dropout(F.relu(self.fc1(x)))
        y = self.fc2(y)
        return y


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super(NormLayer, self).__init__()
        self.size = hidden_dim
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.epsilon = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.epsilon) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = NormLayer(hidden_dim)
        self.norm_2 = NormLayer(hidden_dim)
        self.attn = MultiHeadAttention(heads, hidden_dim, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        y1 = self.norm_1(x)
        y = x + self.dropout_1(self.attn(y1, y1, y1, mask))
        y2 = self.norm_2(y)
        y = y + self.dropout_2(self.ffn(y2))
        return y


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, N, heads, dropout):
        super(Encoder, self).__init__()
        self.num = N
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pe = PositionalEncoder(hidden_dim)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, heads, dropout) for i in range(self.num)])
        self.norm = NormLayer(hidden_dim)

    def forward(self, x, mask):
        y = self.embedding(x)
        x = self.pe(y)
        for i in range(self.num):
            x = self.layers[i](x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super(DecoderLayer, self).__init__()
        self.norm_1 = NormLayer(hidden_dim)
        self.norm_2 = NormLayer(hidden_dim)
        self.norm_3 = NormLayer(hidden_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, hidden_dim, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, hidden_dim, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(self, x, encoder_outputs, source_mask, target_mask):
        x1 = self.norm_1(x)
        y1 = x + self.dropout_1(self.attn_1(x1, x1, x1, target_mask))
        x2 = self.norm_2(y1)
        y1 = y1 + self.dropout_2(self.attn_2(x2, encoder_outputs, encoder_outputs, source_mask))
        x3 = self.norm_3(y1)
        y1 = y1 + self.dropout_3(self.ffn(x3))
        return y1


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, N, heads, dropout):
        super(Decoder, self).__init__()
        self.num = N
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pe = PositionalEncoder(hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, heads, dropout) for i in range(self.num)])
        self.norm = NormLayer(hidden_dim)

    def forward(self, target, encoder_outputs, source_mask, target_mask):
        x = self.embedding(target)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, encoder_outputs, source_mask, target_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, source_vocab, target_vocab, hidden_dim, layer_num, heads, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab, hidden_dim, layer_num, heads, dropout)
        self.decoder = Decoder(target_vocab, hidden_dim, layer_num, heads, dropout)
        self.out = nn.Linear(hidden_dim, target_vocab)

    def forward(self, source, target, source_mask, target_mask):
        encoder_outputs = self.encoder(source, source_mask)
        decoder_outputs = self.decoder(target, encoder_outputs, source_mask, target_mask)
        output = self.out(decoder_outputs)
        return output


if __name__ == '__main__':
    pe = PositionalEncoder(128, 80)
    pe = pe.cuda()
    x = torch.randn([2, 12, 128]).cuda()
    y = pe(x)
    print(y.shape)

    model = MultiHeadAttention(8, 1024)
    model = model.cuda()
    q = torch.randn(2, 12, 128).cuda()
    k = torch.randn(2, 12, 128).cuda()
    v = torch.randn(2, 12, 128).cuda()

    net = FeedForward(128, middle_dim=128)
    net = net.cuda()
    x = torch.randn(10, 128).cuda()
    y = net(x)
    print(y.size())

    net = Transformer(1000, 1000, 128, 3, 8, 0.1)
    source = torch.tensor([[0,1,5,2,11,23,111,90]])
    source_mask = torch.tensor([[1,1,1,1,1,0,0,0]])

    target = torch.tensor([[123, 111, 23, 45, 11, 11]])
    target_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    y = net(source, target, source_mask, target_mask)
    print(y.size())
