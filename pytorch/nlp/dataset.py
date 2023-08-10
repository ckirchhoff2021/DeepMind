import torch
import json
import numpy as np
from torch.utils.data import Dataset


class ThusNewsData(Dataset):
    def __init__(self, data_file):
        super(ThusNewsData, self).__init__()
        self.inputs = list()
        self.targets = list()
        self.initialize(data_file)

    def initialize(self, data_file):
        samples = json.load(open(data_file, 'r'))
        for i in range(14):
            indices = samples[str(i)]
            for idx in indices:
                self.targets.append(i)
                self.inputs.append(idx)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return np.array(self.inputs[item]), self.targets[item]


if __name__ == '__main__':
    train_file = '/home/cx/A100/datas/words/val.json'
    thu = ThusNewsData(train_file)
    x, y = thu[12]
    print(len(x))
    print(thu[0])

