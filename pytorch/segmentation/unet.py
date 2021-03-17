import torch
import torch.nn as nn
from torchvision import transforms, models


def conv_block(in_chn, out_chn):
    return nn.Sequential(
        nn.Conv2d(in_chn, out_chn, 3, padding=1),
        nn.BatchNorm2d(out_chn),
        nn.ReLU(inplace=True),
    
        nn.Conv2d(out_chn, out_chn, 3, padding=1),
        nn.BatchNorm2d(out_chn),
        nn.ReLU(inplace=True)
    )


def deconv_block(in_chn, out_chn):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chn, out_chn, 4, padding=1, stride=2),
        nn.BatchNorm2d(out_chn),
        nn.ReLU(inplace=True)
    )
    
    
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        self.conv_l1 = conv_block(3, 64)
        self.pool_l1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = conv_block(64, 128)
        self.pool_l2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = conv_block(128, 256)
        self.pool_l3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = conv_block(256, 512)
        self.pool_l4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_link = conv_block(512, 1024)
        
        self.up_r4 = deconv_block(1024, 512)
        self.conv_r4 = conv_block(1024, 512)

        self.up_r3 = deconv_block(512, 256)
        self.conv_r3 = conv_block(512, 256)

        self.up_r2 = deconv_block(256, 128)
        self.conv_r2 = conv_block(256, 128)

        self.up_r1 = deconv_block(128, 64)
        self.conv_r1 = conv_block(128, 64)
        
        # self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.classify = nn.Conv2d(64, num_classes, 1)
        
        
    def forward(self, x):
        l1 = self.conv_l1(x)
        p1 = self.pool_l1(l1)

        l2 = self.conv_l2(p1)
        p2 = self.pool_l2(l2)
        
        l3 = self.conv_l3(p2)
        p3 = self.pool_l3(l3)
        
        l4 = self.conv_l4(p3)
        p4 = self.pool_l4(l4)
        
        mid = self.conv_link(p4)
        
        u4 = self.up_r4(mid)
        x4 = torch.cat((l4, u4), dim=1)
        r4 = self.conv_r4(x4)
        
        u3 = self.up_r3(r4)
        x3 = torch.cat((l3, u3), dim=1)
        r3 = self.conv_r3(x3)

        u2 = self.up_r2(r3)
        x2 = torch.cat((l2, u2), dim=1)
        r2 = self.conv_r2(x2)

        u1 = self.up_r1(r2)
        x1 = torch.cat((l1, u1), dim=1)
        r1 = self.conv_r1(x1)
        
        y = self.classify(r1)
        return y
 

def main():
    net = UNet()
    x = torch.randn(1,3,256,256)
    y = net(x)
    print(y.size())



if __name__ == '__main__':
    main()