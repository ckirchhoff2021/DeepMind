import torch
import torch.nn as nn
from vgg import VGG


class FCN32s(nn.Module):
    def __init__(self, num_classes, backbone='vgg'):
        super(FCN32s, self).__init__()
        if backbone == 'vgg':
            self.features = VGG()
            
        # 1/16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 1/8
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 1/4
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 1/2
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 1/1
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(32, num_classes, 1)
        
    def forward(self, x):
        features = self.features(x)
        y16 = self.deconv1(features[4])
        y8 = self.deconv2(y16)
        y4 = self.deconv3(y8)
        y2 = self.deconv4(y4)
        y1 = self.deconv5(y2)
        y_out = self.classifier(y1)
        return y_out
        

class FCN16s(nn.Module):
    def __init__(self, num_classes, backbone='vgg'):
        super(FCN16s, self).__init__()
        if backbone == 'vgg':
            self.features = VGG()
        
        # 1/16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 1/8
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 1/4
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 1/2
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 1/1
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        features = self.features(x)
        y16 = self.deconv1(features[4]) + features[3]
        y8 = self.deconv2(y16)
        y4 = self.deconv3(y8)
        y2 = self.deconv4(y4)
        y1 = self.deconv5(y2)
        y_out = self.classifier(y1)
        return y_out


class FCN8s(nn.Module):
    def __init__(self, num_classes, backbone='vgg'):
        super(FCN8s, self).__init__()
        if backbone == 'vgg':
            self.features = VGG()
        
        # 1/16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 1/8
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 1/4
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 1/2
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 1/1
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        features = self.features(x)
        y16 = self.deconv1(features[4]) + features[3]
        y8 = self.deconv2(y16) + features[2]
        y4 = self.deconv3(y8)
        y2 = self.deconv4(y4)
        y1 = self.deconv5(y2)
        y_out = self.classifier(y1)
        return y_out


def main():
    n1 = FCN32s(10)
    n2 = FCN16s(10)
    n3 = FCN8s(10)
    
    x = torch.randn(1,3,256,256)
    y1 = n1(x)
    print(y1.size())
    
    y2 = n2(x)
    print(y2.size())
    
    y3 = n3(x)
    print(y3.size())


if __name__ == '__main__':
    main()
