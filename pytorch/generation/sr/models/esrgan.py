import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import sr.models.arch_util as arch_util


class ResidualDenseBlock5(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock5, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5(nf, gc)
        self.RDB2 = ResidualDenseBlock5(nf, gc)
        self.RDB3 = ResidualDenseBlock5(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upsample_factor, gc=32):
        super(RRDBNet, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self._make_RRDB_blocks_(nb, nf, gc)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        self.upsample = self._make_upsample_blocks_(nf, upsample_factor, upsample_method='conv')
        self.conv2 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_RRDB_blocks_(self, nb, nf, gc):
        layers = []
        for i in range(nb):
            layers.append(RRDB(nf, gc))
        return nn.Sequential(*layers)
    
    def _make_upsample_blocks_(self, nf, upsample_factor, upsample_method='conv'):
        num = upsample_factor // 2
        layers = list()
        for i in range(num):
            if upsample_method == 'conv':
                layers += [
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ]
            else:
                layers += [
                    nn.Conv2d(nf, nf * 4, 3, 1, 1),
                    nn.PixelShuffle(4),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ]
        return nn.Sequential(*layers)
        

    def forward(self, x):
        fea = self.conv1(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.relu(self.upsample(fea))
        out = self.conv2(fea)
        return out


class ESRGANDiscriminator(nn.Module):
    def __init__(self, num_channel=3):
        super(ESRGANDiscriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.02)
        
        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y_conv1 = self.leaky_relu(self.conv1(x))
        y_conv2 = self.bn2(self.leaky_relu(self.conv2(y_conv1)))
        y_conv3 = self.bn3(self.leaky_relu(self.conv3(y_conv2)))
        y_conv4 = self.bn4(self.leaky_relu(self.conv4(y_conv3)))
        y_conv5 = self.bn5(self.leaky_relu(self.conv5(y_conv4)))
        y_conv6 = self.bn6(self.leaky_relu(self.conv6(y_conv5)))
        y_conv7 = self.bn7(self.leaky_relu(self.conv7(y_conv6)))
        y_conv8 = self.bn8(self.leaky_relu(self.conv8(y_conv7)))
        
        y_avg = self.avg_pool(y_conv8)
        y_avg = y_avg.view(y_avg.size(0), -1)
        y_fc1 = self.leaky_relu(self.fc1(y_avg))
        y_fc2 = self.fc2(y_fc1)
        y_out = self.sigmoid(y_fc2)
        
        return y_out


if __name__ == '__main__':
    net = RRDBNet(3, 3, 64, 8, 2, gc=32)
    print(net)
    
    x = torch.randn(1,3,64,64)
    y = net(x)
    print(y.size())
    
    D = ESRGANDiscriminator()
    y2 = D(y)
    print(y2.size())
