import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetGenerator(nn.Module):
    def __init__(self, in_chn, num_down, nf=64):
        super(UnetGenerator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chn, nf, 3, padding=1, stride=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True)
        )
        self.num_sampling = num_down
        self.down_samples = self.create_down_sampling(nf, num_down)
        self.up_samples = self.create_up_sampling(nf, num_down)
        self.conv2 = nn.Conv2d(nf, in_chn, 3, padding=1, stride=1)

    def create_down_sampling(self, nf, num_down):
        chn = nf
        downs = list()
        for i in range(num_down):
            conv = nn.Sequential(
                nn.Conv2d(chn, chn * 2, 4, padding=1, stride=2),
                nn.BatchNorm2d(chn * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )
            downs.append(conv)
            chn = chn * 2
        return nn.Sequential(*downs)

    def create_up_sampling(self, nf, num_down):
        chn = nf * (2 ** num_down)
        ups = list()
        for i in range(num_down):
            conv = nn.Sequential(
                nn.ConvTranspose2d(chn, chn // 2, 4, padding=1, stride=2),
                nn.BatchNorm2d(chn // 2),
                nn.ReLU(inplace=True)
            )
            ups.append(conv)
            chn = chn // 2
        return nn.Sequential(*ups)


    def forward(self, x):
        y = self.conv1(x)
        downs = list()
        num = self.num_sampling
        for i in range(num):
            block = self.down_samples[i]
            y = block(y)
            downs.append(y)

        for i in range(num):
            block = self.up_samples[i]
            y = block(downs[num-1-i] + y)

        y = self.conv2(y)
        return y


class PixelDiscriminator(nn.Module):
    def __init__(self,in_chn, nf=64):
        super(PixelDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chn, nf, 1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 1, stride=1, padding=0),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(nf * 2, 1, 1, stride=1, padding=0),
            nn.Conv2d(nf * 2, in_chn, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv1(x)
        # y = y.view(y.size(0), -1)
        # return torch.mean(y, dim=1)
        return y


def main():
    G = UnetGenerator(3, 3)
    x = torch.randn(1,3,256,256)
    y = G(x)
    print(y.size())



if __name__ == '__main__':
    main()

