import os
import torch
import torch.nn as nn
from tqdm import tqdm

from collections import OrderedDict
from segmentation.fcn import FCN8s
from segmentation.unet import UNet

import visdom
import numpy as np
from PIL import Image
from torchvision import transforms, models, datasets
import cv2


image_transform = transforms.Compose([
    # transforms.Resize((256,256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor_transform = transforms.Compose(
    [
        transforms.ToPILImage()
    ]
)

def load_fcn8s():
    net = FCN8s(2)
    state = torch.load('data/fcn8s.pth', map_location='cpu')
    state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        state_dict[name] = v
    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_unet():
    net = UNet()
    state = torch.load('data/unet.pth', map_location='cpu')
    state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        state_dict[name] = v
    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_fcn_resnet():
    net = models.segmentation.fcn_resnet50()
    state = torch.load('data/fcn_resnet50.pth', map_location='cpu')
    state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        state_dict[name] = v
    net.load_state_dict(state_dict)
    net.eval()
    return net


class Segmentation:
    def __init__(self):
        self.net = load_fcn8s()
        # self.net = load_unet()
        
    def predict(self, image_file, mask_file):
        id = image_file.split('/')[-1]
        image_data = Image.open(image_file)
        image_tensor = image_transform(image_data)
        x_in = image_tensor.unsqueeze(0)
        y_out = self.net(x_in)
        y_pred = y_out.squeeze().min(0)[1]
        y_pred = y_pred.float()

        s1 = image_data.resize((256, 256))
        s2 = tensor_transform(y_pred)
        s3 = Image.open(mask_file).resize((256,256))
        
        image_compose = Image.new('RGB', (768, 256), 0)
        image_compose.paste(s1, (0,0,256,256))
        image_compose.paste(s3, (256,0, 512, 256))
        image_compose.paste(s2, (512, 0, 768, 256))
        image_compose.save('output/' + id)
    
    
    def batch_predict(self):
        test_image_folder = 'data/test/images'
        test_mask_folder = 'data/test/masks'
        
        test_images = os.listdir(test_image_folder)
        for data in tqdm(test_images):
            if not data.endswith('.jpg'):
                continue
            image_file = os.path.join(test_image_folder, data)
            mask_file = os.path.join(test_mask_folder, data)
            self.predict(image_file, mask_file)
        

def main():
    p1 = '/Users/ckirchhoff/Downloads/Deepmind/segmentation/data/train/images/0.jpg'
    image_data = cv2.imread(p1)
    m1 = '/Users/ckirchhoff/Downloads/Deepmind/segmentation/data/train/masks/0.jpg'
    mask_data = cv2.imread(m1)
    print(mask_data.shape)
    
    vis = visdom.Visdom(env='Test')
    x = torch.arange(1, 100, 0.01)
    y1 = torch.sin(x)
    y2 = torch.cos(x)
    vis.line(X=x,Y=np.column_stack((y1.numpy(), y2.numpy())), win='sinx', opts={'title':'curves'})
    vis.images(torch.randn(36, 3, 64, 64).numpy(), nrow=6, win='images', opts={'title': 'images'})
    
    hstack = np.hstack((image_data, mask_data))
    pil_data = Image.open(p1)
    pil_mask = Image.open(m1)
    
    data_tensor = image_transform(pil_data)
    
    x2 = np.random.randn(100, 2)
    y2 = np.random.randn(100)
    y2 = (y2 > 0).astype(np.int) + 1
    vis.scatter(X=x2, Y=y2, win='scatters', opts={'title': 'scatter', 'legend':['Men', 'Women'], 'markersize':5})

    # cv2.imshow('Test', hstack)
    # cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
    # p1 = '/Users/ckirchhoff/Downloads/Deepmind/segmentation/data/train/images/11.jpg'
    # m1 = '/Users/ckirchhoff/Downloads/Deepmind/segmentation/data/train/masks/11.jpg'
    # seg = Segmentation()
    # seg.predict(p1, m1)
    # seg.batch_predict()
    
    # net = load_fcn_resnet()
    # print(net)
    #
    # voc = datasets.VOCSegmentation('../datas', download=False)
    # data, mask = voc[10]
    # image_tensor = image_transform(data)
    # y_out = net(image_tensor.unsqueeze(0))
    # y_out = y_out['out']
    # y_pred = y_out.max(1)[1]
    # y_pred = y_pred.squeeze()
    #
    # color_map = mask.getpalette()
    # color_map = np.reshape(np.array(color_map), (256, 1, 3)).astype(np.uint8)
    # image = y_pred.numpy().astype(np.uint8)
    # print(image.shape)
    # color_image = Image.fromarray(image)
    #
    # color_image.putpalette(color_map)
    # color_image.show()
    # mask.show()
    # data.show()
    #