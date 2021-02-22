import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        y1 = self.relu1(self.bn1(self.conv1(x)))
        y2 = self.bn2(self.conv2(y1))
        return y2 + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
    
    def forward(self, x):
        return self.shuffler(self.conv(x))


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channel=1, base_filter=64):
        super(Generator, self).__init__()
   
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.residual_blocks = self._make_residual_blocks_(n_residual_blocks, base_filter)
        self.upsample_blocks = self._make_upsample_blocks_(upsample_factor, base_filter)
        
        self.conv2 = nn.Conv2d(base_filter, num_channel, kernel_size=3, stride=1, padding=1)
    
    
    def _make_residual_blocks_(self, num, in_channels):
        layers = list()
        for i in range(num):
            block = ResidualBlock(in_channels)
            layers.append(block)
        return nn.Sequential(*layers)
    
    
    def _make_upsample_blocks_(self, upsample_factor, in_channels):
        counts = upsample_factor // 2
        layers = list()
        for i in range(counts):
            layers += [
                nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        y_conv1 = self.relu1(self.conv1(x))
        y_residual = self.residual_blocks(y_conv1)
        y_upsample = self.upsample_blocks(y_residual)
        y_out = self.conv2(y_upsample)
        return y_out
    


class Discriminator(nn.Module):
    def __init__(self, num_channel=3):
        super(Discriminator, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        
        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        y_conv1 = self.relu(self.conv1(x))
        y_conv2 = self.bn2(self.relu(self.conv2(y_conv1)))
        y_conv3 = self.bn3(self.relu(self.conv3(y_conv2)))
        y_conv4 = self.bn4(self.relu(self.conv4(y_conv3)))
        y_conv5 = self.bn5(self.relu(self.conv5(y_conv4)))
        y_conv6 = self.bn6(self.relu(self.conv6(y_conv5)))
        y_conv7 = self.bn7(self.relu(self.conv7(y_conv6)))
        y_conv8 = self.bn8(self.relu(self.conv8(y_conv7)))

        y_avg = self.avg_pool(y_conv8)
        y_avg = y_avg.view(y_avg.size(0), -1)
        y_fc1 = self.relu(self.fc1(y_avg))
        y_fc2 = self.fc2(y_fc1)
        y_out = self.sigmoid(y_fc2)
        
        return y_out
    

def main():
    G = Generator(10, 4, num_channel=3)
    x = torch.randn(1, 3, 256, 256)
    y = G(x)
    print(y.size())
    
    D = Discriminator(num_channel=3)
    y = D(x)
    print(y.size())


if __name__ == '__main__':
    main()