import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(nets, scale=1):
    if not isinstance(nets, list):
        net = [nets]
    for net in nets:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        initialize_weights([self.conv1, self.conv2], 0.1)
        
    def forward(self, x):
        y1 = self.relu1(self.bn1(self.conv1(x)))
        y2 = self.bn2(self.conv2(y1))
        return x + y2


class ResidualBlock_noBN(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

        initialize_weights([self.conv1, self.conv2], 0.1)
      
    def forward(self, x):
        y1 = self.relu1(self.conv1(x))
        y2 = self.conv2(y1)
        return x + y2


class SRResnet(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channel=3, base_filter=64):
        super(SRResnet, self).__init__()
        self.upsample_factor = upsample_factor
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.02)
        self.conv1 = nn.Conv2d(num_channel, base_filter, 3, stride=1, padding=1)
        
        self.residual_blocks = self._make_residual_blocks_(n_residual_blocks, base_filter)
        self.upsample_blocks = self._make_upsample_blocks_(upsample_factor, base_filter, method='pixelshuffle')
        
        self.conv2 = nn.Conv2d(base_filter, num_channel, 3, stride=1, padding=1)
        
        initialize_weights([self.conv1, self.conv2], 0.1)
        
        
    def _make_residual_blocks_(self, n_residual_blocks, in_channels):
        layers = list()
        for i in range(n_residual_blocks):
            block = ResidualBlock_noBN(in_channels)
            layers.append(block)
        return nn.Sequential(*layers)
    
    
    def _make_upsample_blocks_(self, upsample_factor, in_channels, method='conv'):
        layers = list()
        num = upsample_factor // 2
        for i in range(num):
            if method == 'conv':
                layers += [
                    nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                ]
            else:
                layers += [
                    nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                ]
        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        y_conv1 = self.leaky_relu(self.conv1(x))
        y_residual = self.residual_blocks(y_conv1)
        y_upsample = self.upsample_blocks(y_residual)
        y_conv2 = self.conv2(y_upsample)
        base = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        y_out = y_conv2 + base
        return y_out
    
        
def main():
    x = torch.randn(1,3,400,400)
    net = SRResnet(15, 4)
    print(net)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    main()