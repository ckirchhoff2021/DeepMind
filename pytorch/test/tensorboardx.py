import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import numpy as np


data_root = '/Users/chenx/Desktop/study/data'

class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.conv(x)



def main():
    image_transform = transforms.ToTensor()
    datas = datasets.CIFAR10(root=data_root, train=True, download=True, transform=image_transform)
    data_loader = DataLoader(datas, batch_size=100, shuffle=True)
    tb = SummaryWriter()
    net = TestCNN()

    for index, (data, label) in enumerate(data_loader):
        grid = make_grid(data, nrow=10)
        tb.add_image("images", grid)
        # tb.add_graph(net, data)
        break

    tb.close()


def summary_test():
    writer = SummaryWriter(comment='wave')
    epochs = 100
    for epoch in range(epochs):
        x = epoch
        writer.add_scalar('data/scalar', x * np.sin(x), x)
        # writer.add_scalar('data/scalar', x * np.cos(x), x)

    writer.close()



if __name__ == '__main__':
    # main()

    summary_test()