import torch
import torch.nn as nn
from torchvision import models, transforms


class GraphConv(nn.Module):
    def __init__(self, A, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.adj = A
        self.w = nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = nn.Parameter(torch.randn(output_dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        y = torch.mm(torch.mm(self.adj, x), self.w) + self.b
        y = self.relu(y)
        return y


class GCN(nn.Module):
    def __init__(self, layers, A):
        super(GCN, self).__init__()
        self.adj = A
        self.conv1 = self._make_layers_(layers)

    def _make_layers_(self, layers):
        count = len(layers)
        conv_list = list()
        for i in range(1, count):
            ch1 = layers[i-1]
            ch2 = layers[i]
            conv_list.append(GraphConv(self.adj, ch1, ch2))
        return nn.Sequential(*conv_list)

    def forward(self, x):
        y = self.conv1(x)
        return y


def main():
    A = torch.tensor([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ], dtype=torch.float32)

    net = GraphConv(A, 16, 8)
    x = torch.randn(3, 16)
    y = net(x)
    print(y.size())

    layers = [16, 32, 64, 128, 64]
    net2 = GCN(layers, A)
    y2 = net2(x)
    print(y2.size())



if __name__ == '__main__':
    main()
    # x1 = torch.randn(3, 4)
    # x2 = torch.randn(4, 3)
    # y = torch.mm(x1, x2)
    # print(y.size())
