import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from segmentation.unet import UNet
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models

from collections import OrderedDict

image_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

class DataInstance(Dataset):
    def __init__(self, folder):
        super(DataInstance, self).__init__()
        self.datas = list()
        self.masks = list()
        self.initialize(folder)
        
    def initialize(self, folder):
        image_folder = os.path.join(folder, 'images')
        mask_folder = os.path.join(folder, 'masks')
        image_list = os.listdir(image_folder)
        
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            image_file = os.path.join(image_folder, image)
            mask_file = os.path.join(mask_folder, image)
            self.datas.append(image_file)
            self.masks.append(mask_file)
            
    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, item):
        image_file = self.datas[item]
        mask_file = self.masks[item]
        
        mask_data = np.array(Image.open(mask_file).convert('1').resize((256,256)))
        mask = np.zeros((mask_data.shape[0], mask_data.shape[1], 2), dtype=np.uint8)
        mask[:,:,0] = mask_data
        mask[:,:,1] = ~mask_data
        mask_tensor = transforms.ToTensor()(mask) * 255
        
        image_data = Image.open(image_file)
        image_tensor = image_transform(image_data)
        return image_tensor, mask_tensor
        
        
class VOCSegmentation(Dataset):
    def __init__(self, train=True):
        super(VOCSegmentation, self).__init__()
        self.train = train
        if train:
            image_set = 'train'
        else:
            image_set = 'val'
        self.voc = datasets.VOCSegmentation('../datas', download=False, transform=image_transform, target_transform=mask_transform, image_set=image_set)
        
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, item):
        data, mask = self.voc[item]
        mask = (mask * 255).long()
        mask[mask > 20] = 0
        # mask_tensor = torch.zeros(21, mask.size(1), mask.size(2)).scatter(0, mask, 1)
        return data, mask
        

def load_unet():
    # net = UNet(21)
    net = models.segmentation.fcn_resnet50()
    state = torch.load('data/voc_unet.pth', map_location='cpu')
    state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        state_dict[name] = v
    net.load_state_dict(state_dict)
    net.eval()
    return net
  
    
def main():
    voc = VOCSegmentation()
    data, mask = voc[0]
    print(data.size())
    print(mask.size())
    
    net = UNet(21)
    criterion = nn.CrossEntropyLoss()
    a = data.unsqueeze(0)
  
    y = net(a)
    loss = criterion(y, mask)
    print(loss.item())
    print(y.size())

def voc():
    voc = datasets.VOCSegmentation('../datas', download=False)
    indices = set()
    for i in tqdm(range(len(voc))):
        image_data, mask = voc[i]
        data = np.array(mask)
        a = set(np.unique(data))
        indices = indices | a
    print(indices)
    
    
def voc_seg_test():
    net = load_unet()
    voc =  datasets.VOCSegmentation('../datas', download=False, transform=image_transform)
    data, mask = voc[12]
    y_out = net(data.unsqueeze(0))
    y_out = y_out['out']
    y_pred = y_out.max(1)[1]
    y_pred = y_pred.squeeze()
    
    color_map = mask.getpalette()
    color_map = np.reshape(np.array(color_map), (256,1,3)).astype(np.uint8)
    # print(color_map)
    
    image = y_pred.numpy().astype(np.uint8)
    print(image.shape)
    color_image = Image.fromarray(image)
   
    color_image.putpalette(color_map)
    color_image.show()
    print(color_image.mode)
    
    mask = mask.resize((256,256))
    mask.show()
    
    
if __name__ == '__main__':
    # main()
    # voc()
    # voc_seg_test()
    
    net = models.segmentation.fcn_resnet50()
    # net = models.segmentation.deeplabv3_resnet50()
    print(net)
    
    x = torch.randn(10, 3, 512, 257)
    y = net(x)
    print(y)
    
    
    # f1 = nn.Conv2d(3, 10, 3, stride=1, padding=1, dilation=2)
    # x1 = torch.randn(1,3,128,128)
    # y1 = f1(x1)
    # print(y1.size())
    
    