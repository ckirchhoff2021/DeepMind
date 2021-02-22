import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_image(image_file):
    image_data = Image.open(image_file).convert('YCbCr')
    y, _, _ = image_data.split()
    return y


class DatasetFromFolder(Dataset):
    def __init__(self, image_folder, crop_size, upscale_factor=2, train=True):
        super(DatasetFromFolder, self).__init__()
        self.train = train
        self.datas = list()
        self.targets = list()
        
        self.crop_tool = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_size, crop_size))
            ]
        )
        
        self.input_transform = transforms.Compose(
            [
                transforms.Resize(((crop_size // upscale_factor),(crop_size // upscale_factor))),
                transforms.ToTensor()
            ]
        )
        
        self.target_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        
        self.initialize(image_folder)
    
    
    def initialize(self, image_folder):
        data_list = os.listdir(image_folder)
        for data in data_list:
            if not data.endswith('.jpg') and not data.endswith('.png'):
                continue
            image_file = os.path.join(image_folder, data)
            self.datas.append(image_file)


    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        image_file = self.datas[item]
        # image_data = load_image(image_file)
        image_data = Image.open(image_file)
        image_crop = self.crop_tool(image_data)
        target = image_crop.copy()

        image_data = self.input_transform(image_crop)
        target = self.target_transform(target)
      
        return image_data, target
    


def main():
    A = DatasetFromFolder('../datas/BSDS300/images/train/', 256, train=False)
    x, y = A[12]
    y.show()
    

if __name__ == '__main__':
    main()