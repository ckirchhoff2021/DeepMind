import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets

cuda = torch.cuda.is_available()

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, ], std=[0.2,])
])


augmentation_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


common_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def mnist_dataset():
    print('--- mnist dataset -- ')
    folder = '../../../datas'
    train_datas = datasets.MNIST(folder, train=True, transform=mnist_transform)
    test_datas = datasets.MNIST(folder, train=False, transform=mnist_transform)
    return train_datas, test_datas


class DatasetInstance(Dataset):
    def __init__(self, sample_file, train=False):
        self.datas = list()
        self.targets = list()
        self.train = train
        self.initialize(sample_file)

    def __getitem__(self, item):
        image_file = self.datas[item]
        image_data = Image.open(image_file).convert('RGB')
        if self.train:
            image_tensor = augmentation_transform(image_data)
        else:
            image_tensor = common_transform(image_data)
        return image_tensor, self.targets[item]

    def __len__(self):
        return len(self.datas)

    def initialize(self, sample_file):
        datas = json.load(open(sample_file, 'r'))
        for value in datas:
            image_file = value[0]
            int_label = value[1]
            self.datas.append(image_file)
            self.targets.append(int_label)

    def reweighting(self, beta):
        counts = len(self.targets)
        classes = max(self.targets) + 1
        samples = [0] * classes
        for i in range(counts):
            samples[self.targets[i]] += 1
        samples = np.array(samples)
        weights = (1 - beta) / (1 - beta ** samples)
        weights = weights / np.sum(weights)
        return weights
