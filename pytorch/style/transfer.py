import copy
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets

from style.util import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VGG16(nn.Module):
    def __init__(self, required_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=True).features
        vgg_pretrained = vgg_pretrained.eval()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained[x])

        if not required_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        y = self.slice1(x)
        h1 = y
        y = self.slice2(y)
        h2 = y
        y = self.slice3(y)
        h3 = y
        y = self.slice4(y)
        h4 = y
        y = self.slice5(y)
        h5 = y
        vgg_features = namedtuple('vgg_features', ['h1', 'h2', 'h3', 'h4', 'h5'])
        features = vgg_features(h1, h2, h3, h4, h5)
        return features


def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a, b, c * d)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (b * c * d)
    return gram


def compute_content_loss(source, target):
    loss = F.mse_loss(source, target)
    return loss


def compute_style_loss(source, target):
    A = gram_matrix(source)
    B = gram_matrix(target)
    loss = F.mse_loss(A, B)
    return loss



def run():
    # content_file = '/Users/chenxiang/Desktop/images/dancing.jpg'
    content_file = "/Users/chenxiang/Desktop/images/bro.jpg"
    style_file = '/Users/chenxiang/Desktop/images/picasso.jpg'

    content_transform = transforms.Compose([
        transforms.CenterCrop((2736,2736)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    net = VGG16()
    content_image = Image.open(content_file)
    content_tensor = content_transform(content_image)
    content_tensor = content_tensor.unsqueeze(0)
    content_tensor.requires_grad = False

    style_image = Image.open(style_file)
    style_tensor = image_transform(style_image)
    style_tensor = style_tensor.unsqueeze(0)
    style_tensor.requires_grad = False

    input_image = Image.open(content_file)
    input_tensor = content_transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    opt = optimizer.Adam([input_tensor.requires_grad_()], lr=0.1)
    epochs = 300

    style_feature = net(style_tensor)
    s1, s2, s3, s4, s5 = style_feature

    content_feature = net(content_tensor)
    c1, c2, c3, c4, c5 = content_feature

    for epoch in range(epochs):
        input_features = net(input_tensor)
        v1, v2, v3, v4, v5 = input_features
        s1_loss = compute_style_loss(s1, v1)
        s2_loss = compute_style_loss(s2, v2)
        s3_loss = compute_style_loss(s3, v3)
        s4_loss = compute_style_loss(s4, v4)
        s5_loss = compute_style_loss(s5, v5)

        style_loss = (s1_loss + s2_loss + s3_loss + s4_loss + s5_loss) * 1e6
        content_loss = compute_content_loss(c2, v2) * 1.0
        loss = style_loss + content_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('==> Epoch: [%d]/[%d], loss = %f, style_loss = %f, content_loss = %f' % (epoch, epochs, loss.item(), style_loss.item(), content_loss.item()))

    imshow(input_tensor, title='output')
    plt.show()

    out = recover_image(input_tensor)
    out_image = Image.fromarray(out)
    out_image.save('bro.png')


def main():
    run()


if __name__ == '__main__':
    main()


    # A = torch.randn(1,3,128,128)
    # B = gram_matrix(A)
    # print(B.size())




