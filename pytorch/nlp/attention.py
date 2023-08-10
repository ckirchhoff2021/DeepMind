import torch
import torch.nn as nn
import time


class SelfAttention(nn.Module):
    def __init__(self, token_dim, heads=10):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.wq_list = list()
        self.wk_list = list()
        self.wv_list = list()
        for i in range(heads):
            self.wq_list.append(nn.Parameter(torch.randn(token_dim, token_dim).cuda()))
            self.wk_list.append(nn.Parameter(torch.randn(token_dim, token_dim).cuda()))
            self.wv_list.append(nn.Parameter(torch.randn(token_dim, token_dim).cuda()))
        self.scale = token_dim ** 0.5
        self.wo = nn.Parameter(torch.randn(token_dim * heads, token_dim).cuda())

    def expand(self, w, batch_size):
        w1 = w.unsqueeze(0)
        w1 = w1.expand(batch_size, -1, -1)
        return w1

    def forward(self, x):
        batch_size = x.size(0)
        out_list = []
        for i in range(self.heads):
            wq = self.expand(self.wq_list[i], batch_size)
            Q = torch.bmm(x, wq)
            wk = self.expand(self.wk_list[i], batch_size)
            K = torch.bmm(x, wk)
            wv = self.expand(self.wk_list[i], batch_size)
            V = torch.bmm(x, wv)
            KT = K.transpose(2,1)
            S = torch.bmm(Q, KT) / self.scale
            P = torch.softmax(S, dim=-1)
            O = torch.bmm(P, V)
            out_list.append(O)
        out = torch.cat(out_list, dim=-1)
        wo = self.expand(self.wo, batch_size)
        output = torch.bmm(out, wo)
        return output


def group_attention(Q,K,V,groups,scale):
    Qs = torch.chunk(Q, groups, dim=1)
    Ks = torch.chunk(K, groups, dim=1)
    Vs = torch.chunk(V, groups, dim=1)
    Os = []
    for i in range(groups):
        j = (i + 1) % groups
        KsT = Ks[j].transpose(1,2)
        Si = torch.bmm(Qs[i], KsT)
        Pi = torch.softmax(Si, dim=-1)
        Oi = torch.bmm(Pi, Vs[j])
        Os.append(Oi)
    O = torch.cat(Os, dim=1)
    return O


class GroupAttention(nn.Module):
    def __init__(self, token_dim, groups=4, heads=10):
        super(GroupAttention, self).__init__()
        self.heads = heads
        self.groups = groups
        self.wq_list = list()
        self.wk_list = list()
        self.wv_list = list()
        for i in range(heads):
            self.wq_list.append(nn.Parameter(torch.randn(token_dim, token_dim).cuda()))
            self.wk_list.append(nn.Parameter(torch.randn(token_dim, token_dim).cuda()))
            self.wv_list.append(nn.Parameter(torch.randn(token_dim, token_dim).cuda()))
        self.scale = token_dim ** 0.5
        self.wo = nn.Linear(token_dim * heads, token_dim, bias=False)

    def expand(self, w, batch_size):
        w1 = w.unsqueeze(0)
        w1 = w1.expand(batch_size, -1, -1)
        return w1

    def forward(self, x):
        batch_size = x.size(0)
        out_list = []
        for i in range(self.heads):
            wq = self.expand(self.wq_list[i], batch_size)
            Q = torch.bmm(x, wq)
            wk = self.expand(self.wk_list[i], batch_size)
            K = torch.bmm(x, wk)
            wv = self.expand(self.wk_list[i], batch_size)
            V = torch.bmm(x, wv)
            O = group_attention(Q, K, V, self.groups, self.scale)
            out_list.append(O)
        out = torch.cat(out_list, dim=-1)
        output = self.wo(out)
        return output


class AttentionModel(nn.Module):
    def __init__(self, token_dim=8):
        super(AttentionModel, self).__init__()
        self.embeddings = nn.Embedding(100000, token_dim)
        self.encoder = SelfAttention(token_dim, heads=4)
        self.sequence_length = 512
        fnn_dim = self.sequence_length * token_dim
        self.fnn = nn.Sequential(
            nn.Linear(fnn_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 14)
        )

    def forward(self, x):
        inputs = self.embeddings(x)
        x1 = self.encoder(inputs)
        x2 = x1.view(x1.size(0), -1)
        out = self.fnn(x2)
        return out


class GroupAttentionModel(nn.Module):
    def __init__(self, token_dim=8, groups=4):
        super(GroupAttentionModel, self).__init__()
        self.embeddings = nn.Embedding(100000, token_dim)
        self.encoder = GroupAttention(token_dim, heads=4, groups=groups)
        self.sequence_length = 512
        fnn_dim = self.sequence_length * token_dim
        self.fnn = nn.Sequential(
            nn.Linear(fnn_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 14)
        )

    def forward(self, x):
        inputs = self.embeddings(x)
        x1 = self.encoder(inputs)
        x2 = x1.view(x1.size(0), -1)
        out = self.fnn(x2)
        return out


if __name__ == '__main__':
    model = AttentionModel(token_dim=8)
    model = model.cuda()
    N = 10000
    x = torch.randint(0, 10000, (16, 512)).cuda()
    k1 = time.time()
    for i in range(N):
        y = model(x)
    k2 = time.time()
    print('==> E2E attention: ', (k2 - k1) / N)
    print(y.shape)

    net = GroupAttentionModel(token_dim=8)
    net = net.cuda()
    x = torch.randint(0, 10000, (16, 512)).cuda()
    k1 = time.time()
    for i in range(N):
        y = net(x)
    k2 = time.time()
    print('==> E2E group attention: ', (k2 - k1) / N)
    print(y.shape)
