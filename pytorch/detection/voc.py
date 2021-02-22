from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


label_list = [
    'cat', 'horse', 'bottle', 'sofa', 'dog', 'chair', 'diningtable', 'sheep', 'bird',
    'aeroplane', 'motorbike', 'bus', 'car', 'train', 'tvmonitor', 'cow', 'pottedplant',
    'bicycle', 'person', 'boat'
]

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

class VocDataset(Dataset):
    def __init__(self, train=True, image_size=448):
        super(VocDataset, self).__init__()
        self.datas = list()
        self.targets = list()
        self.image_size = image_size
        self.train = train
        self.initialize()
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        return self.datas[item], self.targets[item]
        
    def initialize(self):
        voc_datas = datasets.VOCDetection('../datas', download=False)
        data_num = len(voc_datas)
        seg = int(0.001* data_num)
        if self.train:
            indices = list(range(0, data_num-seg))
        else:
            indices = list(range(data_num-seg, data_num))
        
        for index in indices:
            data = voc_datas[index]
            image_data = data[0]
            data_size = image_data.size
            
            image_data = image_data.resize((self.image_size, self.image_size))
            image_tensor = train_transforms(image_data)
            self.datas.append(image_tensor)
            target = self.generate_label(data[1], data_size)
            self.targets.append(target)
    
    
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
            label[row][col][20:22] = [1, 0]
            label[row][col][22:26] = np.array([cx, cy, w, h]) / self.image_size
        
        return torch.from_numpy(label)


def main():
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
   

if __name__ == '__main__':
    main()