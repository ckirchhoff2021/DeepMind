import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(out_num=10, mid_dim=128, pretrained=False):
    net = models.resnet18(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_in_num, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


def resnet34(out_num=10, mid_dim=128, pretrained=False):
    net = models.resnet34(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_in_num, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


def resnet50(out_num=10, mid_dim=128, pretrained=False):
    net = models.resnet50(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_in_num, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


def resnet101(out_num=10, mid_dim=128, pretrained=False):
    net = models.resnet101(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_in_num, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


def resnet152(out_num=10, mid_dim=128, pretrained=False):
    net = models.resnet152(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_in_num, mid_dim),
        nn.Linear(mid_dim, out_num)
    )
    return net


if __name__ == '__main__':
    net = resnet50()
    modules = list(net.children())
    fc = modules[-1]
    
    y1 = nn.Sequential(*modules[:-1])
    modules2 = modules[:-1] + [fc[0]]
    net2 = nn.Sequential(*modules2)
    print(net2)
    
    x = torch.randn(1,3,224,224)
    y = y1(x)
    print(y.size())
    
    