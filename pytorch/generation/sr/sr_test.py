import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from collections import OrderedDict
from models.srgan import Generator, Discriminator
from torchvision import models, transforms

from models.esrgan import RRDBNet
from models.discriminator_vgg import Discriminator_VGG_128

def load_SRGAN():
    G = Generator(n_residual_blocks=16, upsample_factor=2)
    D = Discriminator()
    state = torch.load('output/srgan.pth', map_location='cpu')
    
    state_dict = OrderedDict()
    for k, v in state['G'].items():
        name = k[7:]
        state_dict[name] = v
    G.load_state_dict(state_dict)
    G.eval()

    state_dict = OrderedDict()
    for k, v in state['D'].items():
        name = k[7:]
        state_dict[name] = v
    D.load_state_dict(state_dict)
    D.eval()
    
    return G, D


def load_ESRGAN():
    D = Discriminator_VGG_128(3, 64)
    G = RRDBNet(3, 3, 64, 23, gc=32)
    state = torch.load('output/esrgan.pth', map_location='cpu')
    
    state_dict = OrderedDict()
    for k, v in state['G'].items():
        name = k[7:]
        state_dict[name] = v
    G.load_state_dict(state_dict)
    G.eval()
    
    state_dict = OrderedDict()
    for k, v in state['D'].items():
        name = k[7:]
        state_dict[name] = v
    D.load_state_dict(state_dict)
    D.eval()
    
    return G, D
 
    

def main():
    G, D = load_SRGAN()
    
    image_path = 'config/baboon.png'
    image_data = Image.open(image_path).convert('YCbCr')
    y, cb, cr = image_data.split()
    
    print(y.mode)
    
    x1 = transforms.ToTensor()(y)
    x2 = x1.unsqueeze(0)
    y2 = G(x2)
    
    y_pred = y2.squeeze().data.numpy()
    print(y_pred.shape)
    y_out = y_pred * 255
    y_out = y_out.clip(0, 255).astype(np.uint8)

    y_image = Image.fromarray(y_out, mode='L')
    print(y_image.size)
    y_cb = cb.resize(y_image.size, Image.BICUBIC)
    y_cr = cr.resize(y_image.size, Image.BICUBIC)

    y_image = Image.merge('YCbCr', [y_image, y_cb, y_cr]).convert('RGB')
    y_image.show()
    

def test():
    G, D = load_ESRGAN()
    image_path = 'config/baboon.png'
    image_data = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image_data)
    x2 = image_tensor.unsqueeze(0)
    y1 = G(x2)
    print(y1.size())
    
    y2 = transforms.ToPILImage()(y1[0])
    y2.show()
    
    


if __name__ == '__main__':
    # main()
    
    # net = RRDBNet(3, 3, 64, 23, gc=32)
    # x = torch.randn(1,3,256,256)
    # y = net(x)
    # print(y.size())

    # D = Discriminator_VGG_128(3, 64)
    # x2 = torch.randn(1,3,128,128)
    # y2 = D(x2)
    # print(y2.size())
    
    # test()

    image_path = 'config/baboon.png'
    image_data = Image.open(image_path)
    w, h = image_data.size
    x2 = image_data.resize((w * 2, h * 2), Image.BICUBIC)
    x3 = image_data.resize((w * 2, h * 2), Image.BILINEAR)
    x2.show()
    x3.show()
    