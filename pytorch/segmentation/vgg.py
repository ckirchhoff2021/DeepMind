import os
import torch
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        
        # conv 1/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        if pretrained:
            pretrained_model = models.vgg16(pretrained=True)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = dict()
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)
    
            
    def forward(self, x):
        y1 = self.conv1(x)
        pool1 = y1
        
        y2 = self.conv2(y1)
        pool2 = y2
        
        y3 = self.conv3(y2)
        pool3 = y3
        
        y4 = self.conv4(y3)
        pool4 = y4
        
        y5 = self.conv5(y4)
        pool5 = y5
        return pool1, pool2, pool3, pool4, pool5

def main():
    net = VGG()
    print(net)
    
    x = torch.randn(1,3,256,256)
    y1, y2, y3, y4, y5 = net(x)
    print(y1.size())
    print(y2.size())
    print(y3.size())
    print(y4.size())
    print(y5.size())
    

if __name__ == '__main__':
    main()