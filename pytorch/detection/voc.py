import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import random


label_list = [
    'cat', 'horse', 'bottle', 'sofa', 'dog', 'chair', 'diningtable', 'sheep', 'bird',
    'aeroplane', 'motorbike', 'bus', 'car', 'train', 'tvmonitor', 'cow', 'pottedplant',
    'bicycle', 'person', 'boat'
]

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

class VocDataset(Dataset):
    def __init__(self, train=True, image_size=448):
        super(VocDataset, self).__init__()
        self.voc_origin = datasets.VOCDetection('/Users/chenx/Desktop/study/data', download=False)
        self.indices = None
        self.image_size = image_size
        self.train = train
        self.initialize()
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        index = self.indices[item]
        image_data, image_annotate = self.voc_origin[index]
        data_size = image_data.size
        image_data = image_data.resize((self.image_size, self.image_size))
        image_tensor = train_transforms(image_data)
        target = self.generate_label(image_annotate, data_size)
        return image_tensor, target

    def initialize(self):
        data_num = len(self.voc_origin)
        indices = list(range(data_num))
        random.shuffle(indices)
        seq = int(0.1* data_num)
        if self.train:
            self.indices = indices[0:data_num-seq]
        else:
            self.indices = indices[data_num-seq:]

    
    def box_transform(self, bbox, data_size):
        w, h = data_size
        x_ratio , y_ratio = self.image_size / w, self.image_size / h
        x_min = int(bbox['xmin']) * x_ratio
        x_max = int(bbox['xmax']) * x_ratio
        y_min = int(bbox['ymin']) * y_ratio
        y_max = int(bbox['ymax']) * y_ratio
        
        bw = int(x_max - x_min)
        bh = int(y_max - y_min)
        center = int((x_min + x_max) * 0.5), int((y_min + y_max) * 0.5)
        return [center[0], center[1], bw, bh]
        
    
    def generate_label(self, label_dict, data_size):
        label = np.zeros((7,7,30))
        objects = label_dict['annotation']['object']
        unit = 448 / 7
        
        obj_list = list()
        if type(objects) == dict:
            obj_list.append(objects)

        if type(objects) == list:
            obj_list.extend(objects)

        for object in obj_list:
            str_label = object['name']
            int_label = label_list.index(str_label)
            
            obj_onehot = np.zeros(20)
            obj_onehot[int_label] = 1
            
            bbox = object['bndbox']
            cx, cy, w, h = self.box_transform(bbox, data_size)
            row, col = int(cx/unit), int(cy/unit)

            label[row][col][:20] = obj_onehot

            if label[row][col][20] == 0:
                label[row][col][20] = 1
                label[row][col][22:26] = np.array([cx, cy, w, h]) / self.image_size
            else:
                label[row][col][21] = 1
                label[row][col][26:30] = np.array([cx, cy, w, h]) / self.image_size
        
        return torch.from_numpy(label)


def main():
    '''
    voc = datasets.VOCSegmentation('../datas', download=False, image_set='train')
    print(len(voc))
    image_data, mask = voc[2]
    
    color_map = mask.getpalette()
    color_map = np.array(color_map)
    color_palette = np.reshape(color_map, (256, 1, 3)).astype(np.uint8)
    colors = np.zeros((32, 21 * 32), np.uint8)
    for i in range(21):
        colors[:, 32*i:(i+1)*32] = i
    color_images = Image.fromarray(colors)
    color_images.putpalette(color_palette)
    color_images.show()
    '''

    datas = VocDataset()
    data, labels = datas[1]
    mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
    std = np.array([[[0.229]], [[0.224]], [[0.225]]])
    data2 = data.numpy() * std + mean
    data_reverse = torch.from_numpy(data2.astype(np.float32))
    data = transforms.ToPILImage()(data_reverse)

    draw = ImageDraw.Draw(data)
    for i in range(7):
        for j in range(7):
            annotates = labels[i][j].numpy()
            if annotates[20] == 1:
                x,y,w,h = annotates[22:26]
                x1, y1 = int(448 * x - 224 * w), int(448 * y - 224 * h)
                x2, y2 = int(448 * x + 224 * w), int(448 * y + 224 * h)
                draw.rectangle((x1,y1,x2,y2), outline='red')

            if annotates[21] == 1:
                x,y,w,h = annotates[26:30]
                x1, y1 = int(448 * x - 224 * w), int(448 * y - 224 * h)
                x2, y2 = int(448 * x + 224 * w), int(448 * y + 224 * h)
                draw.rectangle((x1, y1, x2, y2), outline='red')

    data.show()
   

if __name__ == '__main__':
    main()
