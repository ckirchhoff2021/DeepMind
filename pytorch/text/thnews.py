import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import *
import numpy as np


class THUDatas(Dataset):
    def __init__(self, train=False):
        super(THUDatas, self).__init__()
        self.train = train
        self.inputs = None
        self.targets = None
        self.masks = None
        self.initialize()

    def initialize(self):
        xs = np.load('datas/x.npy')
        ys = np.load('datas/y.npy')
        masks = np.load('datas/mask.npy')
        splits = np.load('datas/trainval_split.npy')

        if self.train:
            indices = np.where(splits == True)
        else:
            indices = np.where(splits == False)

        self.inputs = xs[indices]
        self.targets = ys[indices]
        self.masks = masks[indices]

        print(self.inputs.shape)
        print(self.targets.shape)
        print(self.masks.shape)


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):
        x = torch.tensor(self.inputs[index])
        y = torch.tensor(self.targets[index])
        mask = torch.tensor(self.masks[index])
        return x, mask, y


def main():
    d = THUDatas(train=True)
    x, y, m = d[10]
    print(x.size())
    print(y.size())
    print(m.size())



if __name__ == '__main__':
    main()