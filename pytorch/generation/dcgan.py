import torch
import torch.nn as nn


batch_size = 128
image_size = 64
nc = 3
nz = 100
num_epochs = 10
learning_rate = 0.0002
ngf = 64
ndf = 64


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            # 128 x 100 x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # 128 x 512 x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            # 128 x 256 x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # 128 x 128 x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # 128 x 64 x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 128 x 3 x 64 x 64
        )

    def forward(self, x):
        out = self.gen(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            # 128 x 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 64 x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 128 x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x256 x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 512 x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4),
            # 128 x 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.disc(x)
        out = out.view(out.size(0), -1)
        return out


def main():
    D = Discriminator()
    x = torch.rand(128, 3, 64, 64)
    y = D(x)
    print(y.size())

    G = Generator()
    g_x = torch.rand(128, 100, 1, 1)
    g_y = G(g_x)
    print(g_y.size())


if __name__ == '__main__':
    main()