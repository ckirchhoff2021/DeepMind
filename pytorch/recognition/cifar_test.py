import os
from common_path import *
from models.resnet import *
from train import start_build
from datasets import CommonDataset


def main():
    root = ''
    model_file = os.path.join(output_path, 'cls.pth')
    net = resnet50(pretrained=True, out_num=100)
    dataset = CommonDataset(root)
    train_datas, test_datas = dataset.cifar100()
    start_build(train_datas, test_datas, net, 128, 30, model_file)


if __name__ == '__main__':
    main()

