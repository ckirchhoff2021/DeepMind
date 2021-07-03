import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optimizer
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from cls_train import *

root = '/Users/chenxiang/Downloads/KTH'
str_labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

image_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class KTHDataset(Dataset):
    def __init__(self, data_json):
        super(KTHDataset, self).__init__()
        # self.preprocess()
        self.datas = json.load(open(data_json, 'r'))
        self.n_clip = 32


    def preprocess(self):
        for label in str_labels:
            str_folder = os.path.join(root, label)
            video_list = os.listdir(str_folder)
            print('==> %s : %d' % (label, len(video_list)))
            for file in tqdm(video_list):
                video_file = os.path.join(str_folder, file)
                self.extract_frames(video_file, label)


    def extract_frames(self, video_file, str_label):
        extract_folder = '/Users/chenxiang/Downloads/dataset/KTH'
        str_video = video_file.split('/')[-1].split('.')[0]
        label_folder = os.path.join(extract_folder,  str_label)
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)

        pic_folder = os.path.join(label_folder, str_video)
        if not os.path.exists(pic_folder):
            os.mkdir(pic_folder)

        capture = cv2.VideoCapture(video_file)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        # frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # frame_fps = capture.get(cv2.CAP_PROP_FPS)

        retaining = True
        count = 0
        while retaining and count < frame_count:
            retaining, frame = capture.read()
            if frame is None:
                continue
            cv2.imwrite(os.path.join(pic_folder, '{}_{}.jpg'.format(str_video, count)), frame)
            count += 1

        capture.release()


    def initialize(self):
        image_root = '/Users/chenxiang/Downloads/dataset/KTH'
        train_datas = list()
        test_datas = list()
        for index, label in enumerate(str_labels):
            image_folder = os.path.join(image_root, label)
            sub_folder = os.listdir(image_folder)
            print('{}: {}'.format(label, len(sub_folder)))
            random.shuffle(sub_folder)
            count = len(sub_folder)
            pos = int(0.1 * count)
            for i in range(count):
                data_folder = os.path.join(image_folder, sub_folder[i])
                if not os.path.isdir(data_folder):
                    continue
                if i < pos:
                    test_datas.append([data_folder, index])
                else:
                    train_datas.append([data_folder, index])

        print('==> train datas: ', len(train_datas))
        print('==> test datas: ', len(test_datas))
        random.shuffle(train_datas)
        random.shuffle(test_datas)

        with open(os.path.join(image_root, 'train.json'), 'w') as f:
            json.dump(train_datas, f, ensure_ascii=False)

        with open(os.path.join(image_root, 'test.json'), 'w') as f:
            json.dump(test_datas, f, ensure_ascii=False)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        folder, label = self.datas[index]
        image_files = os.listdir(folder)
        if os.path.exists(os.path.join(folder, '.DS_Store')):
            os.remove(os.path.join(folder, '.DS_Store'))

        image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        tensor_list = list()
        for file in image_files:
            image_data = Image.open(os.path.join(folder, file))
            image_tensor = image_transform(image_data)
            image_tensor = image_tensor.unsqueeze(1)
            tensor_list.append(image_tensor)
        video_tensor = torch.cat(tensor_list, 1)
        pos = np.random.randint(video_tensor.size(1) - self.n_clip)
        return video_tensor[:,pos:pos+self.n_clip,:,:], label



class C3DNet(nn.Module):
    def __init__(self):
        super(C3DNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(512, 512, 3, stride=1, padding=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1280, 6)
        )

    def forward(self, x):
        y1 = self.feature(x)
        y1 = y1.view(y1.size(0), -1)
        y2 = self.fc(y1)
        return y2



class KTHFrameDataset(nn.Module):
    def __init__(self, data_json):
        super(KTHFrameDataset, self).__init__()
        self.datas = json.load(open(data_json, 'r'))
        self.n_seg = 3


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        folder, label = self.datas[index]
        image_files = os.listdir(folder)
        if os.path.exists(os.path.join(folder, '.DS_Store')):
            os.remove(os.path.join(folder, '.DS_Store'))

        image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        ncount = len(image_files)
        pos = ncount // 3
        heads = image_files[:pos]
        mids = image_files[pos:pos * 2]
        tails = image_files[pos * 2:]

        k1 = np.random.randint(len(heads))
        k2 = np.random.randint(len(mids))
        k3 = np.random.randint(len(tails))

        x1 = Image.open(os.path.join(folder, heads[k1]))
        x2 = Image.open(os.path.join(folder, mids[k2]))
        x3 = Image.open(os.path.join(folder, tails[k3]))

        x1 = image_transform(x1)
        x2 = image_transform(x2)
        x3 = image_transform(x3)

        inputs = torch.cat([x1, x2, x3], dim=1)
        return inputs, label



class TSN(nn.Module):
    def __init__(self):
        super(TSN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )


    def forward(self, x):
        pos = x.size(2) // 3
        x1 = x[:,:, :pos, :]
        x2 = x[:,:,pos:pos * 2,:]
        x3 = x[:,:,pos * 2:,:]

        y1 = self.feature(x1)
        y2 = self.feature(x2)
        y3 = self.feature(x3)

        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)
        y3 = y3.view(y3.size(0), -1)

        f1 = self.fc(y1)
        f2 = self.fc(y2)
        f3 = self.fc(y3)

        output = (f1 + f2 + f3) / 3
        return output


def main():
    KTH = KTHFrameDataset('/Users/chenxiang/Downloads/dataset/KTH/train.json')
    x, y = KTH[12]
    net = TSN()
    x = x.unsqueeze(0)
    print(x.size())
    y = net(x)
    print(y.size())




def train():
    # net = C3DNet()
    net = TSN()
    train_datas= KTHFrameDataset('/Users/chenxiang/Downloads/dataset/KTH/train.json')
    test_datas = KTHFrameDataset('/Users/chenxiang/Downloads/dataset/KTH/test.json')

    start_train(net, train_datas, test_datas, 30, 0.005, 'summary', 'output/video_cls.pth')


if __name__ == '__main__':
    # main()
    train()