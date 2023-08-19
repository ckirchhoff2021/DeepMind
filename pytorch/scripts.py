import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from framework.classify import TrainInstance

data_folder = '/home/cx/A100/datas/cv'


class ClassifyTest(nn.Module):
    def __init__(self):
        super(ClassifyTest, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x1 = x.view(x.size(0), -1)
        y = self.layer(x1)
        return y

def main():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_set = datasets.MNIST(data_folder, train=True, download=False, transform=train_transforms)
    test_set = datasets.MNIST(data_folder, train=False, download=False, transform=train_transforms)

    x0, y0 = train_set[0]
    print(x0.shape)
    summary_folder = 'output/summary'
    checkpoints = 'output/test.pt'
    log_file = 'output/train.log'
    train_instance = TrainInstance(train_set, test_set, epochs=5, device_ids=[0,1,2,3],
                                   train_batch_size=512, test_batch_size=256,
                                   summary_folder=summary_folder, checkpoints=checkpoints,
                                   log_file=log_file)
    model = ClassifyTest()
    train_instance.run(model)

if __name__ == '__main__':
    main()
