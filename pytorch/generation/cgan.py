import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_fc = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.label_fc = nn.Sequential(
            nn.Linear(10, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.feature = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        x_out = self.data_fc(x)
        y_out = self.label_fc(label)
        out = torch.cat((x_out, y_out), 1)
        out = self.feature(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True)
        )

        self.label_fc = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(inplace=True)
        )

        self.feature = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, label):
        x_out = self.data_fc(z)
        y_out = self.label_fc(label)
        out = torch.cat((x_out, y_out), 1)
        out = self.feature(out)
        return out


def main():
    D = Discriminator()
    x = torch.randn(12, 784)
    y = torch.randn(12, 10)
    y_p = D(x, y)
    print(y_p.size())

    G = Generator()
    z = torch.randn(12, 100)
    z_p = G(z, y)
    print(z_p.size())


if __name__ == '__main__':
    main()