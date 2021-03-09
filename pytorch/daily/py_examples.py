import torch
import torch.nn as nn
import torch.optim as optimizer

import misc_utils as utils

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2)
            # nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return y


def main():
    x = torch.randn(100, 2)
    y =  ((x[:, 0] + x[:,1]) > 0).long()
    net = LinearModel()
    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=0.005)
    for epoch in range(40):
        y1 = net(x)
        loss = criterion(y1, y)
        y2 = y1.max(1)[1]
        acc = y2.eq(y).sum().item() / y.size(0)
        print('==> epoch: [%d]/[%d], loss = %f, acc = %f' % (epoch, 10, loss.item(), acc))

        opt.zero_grad()
        loss.backward()
        opt.step()



if __name__ == '__main__':
    # main()
    utils.color_print('test', 2)