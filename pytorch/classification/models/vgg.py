import torch
import torch.nn as nn
from torchvision import models

config = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNet(nn.Module):
    def __init__(self, vgg_type, in_num=3, out_num=10):
        super().__init__()
        self.features = self._make_layers(config[vgg_type], in_num)
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_num)
        )
    
    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, config_list, in_num):
        layers = []
        in_chn = in_num
        for chn in config_list:
            if chn == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_chn, chn, kernel_size=3, padding=1),
                    nn.BatchNorm2d(chn),
                    nn.ReLU(inplace=True)
                ]
                in_chn = chn
        layers += [ nn.AdaptiveAvgPool2d(output_size=(7, 7))]
        return nn.Sequential(*layers)


def vgg13_sketch():
    return VGGNet('VGG13')


def vgg16_sketch():
    return VGGNet('VGG16')


def vgg19_sketch():
    return VGGNet('VGG19')


def vgg13(pretrained=False, mid_dim=128, out_num=10):
    net = models.vgg13_bn(pretrained=pretrained)
    net.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(2048, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


def vgg16(pretrained=False, mid_dim=128, out_num=10):
    net = models.vgg16_bn(pretrained=pretrained)
    net.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(2048, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


def vgg19(pretrained=False, mid_dim=128, out_num=10):
    net = models.vgg19_bn(pretrained=pretrained)
    net.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(2048, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net



if __name__ == '__main__':
    # test()
    net = vgg19()
    print(net)