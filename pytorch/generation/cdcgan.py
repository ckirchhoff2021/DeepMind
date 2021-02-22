import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.data_conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.label_conv = nn.Sequential(
            nn.Conv2d(10, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.feature = nn.Sequential(
            # 2 x 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 2 x 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 2 x 256 x 4 x 4
            nn.Conv2d(512, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        x_out = self.data_conv(x)
        y_out = self.label_conv(label)

        out = torch.cat((x_out, y_out), 1)
        out = self.feature(out)
        out = out.view(out.size(0), -1)
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_conv = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4),
            nn.ReLU(inplace=True)
        )

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(10, 256, 4),
            nn.ReLU(inplace=True)
        )

        self.feature = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, label):
        x_out = self.data_conv(x)
        y_out = self.label_conv(label)
        out = torch.cat((x_out, y_out), 1)
        out = self.feature(out)
        return out


def main():
    x = torch.randn(12, 100, 1, 1)
    y = torch.randn(12, 10, 1, 1)
    G = Generator()
    y_p = G(x, y)
    print(y_p.size())

    D = Discriminator()
    x_d = torch.randn(12, 1, 32, 32)
    y_d = torch.randn(12, 10, 32, 32)
    y_q = D(x_d, y_d)
    print(y_q.size())


if __name__ == "__main__":
    main()