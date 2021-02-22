import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class TransformNet(nn.Module):
    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(3, base, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base * 2, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(base * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 2, base * 4, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )

        self.residuals = nn.Sequential(
            ResidualBlock(base * 4),
            ResidualBlock(base * 4),
            ResidualBlock(base * 4),
            ResidualBlock(base * 4),
            ResidualBlock(base * 4)
        )

        self.up_sample = nn.Sequential(
            nn.Upsample(mode='nearest', scale_factor=2),
            nn.Conv2d(base * 4, base * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base * 2),

            nn.Upsample(mode='nearest', scale_factor=2),
            nn.Conv2d(base * 2, base, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base),

            nn.Conv2d(base, 3, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(3)
        )

    def forward(self, x):
        y = self.down_sample(x)
        y = self.residuals(y)
        y = self.up_sample(y)
        return y


def main():
    net = TransformNet()
    x = torch.randn(1,3,256, 256)
    y = net(x)
    print(y.size())



if __name__ == '__main__':
    main()

