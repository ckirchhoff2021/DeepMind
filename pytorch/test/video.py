import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import os
import cv2
from tqdm import tqdm

root = '/Users/chenxiang/Downloads/KTH'
str_labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


class KTHDataset(Dataset):
    def __init__(self):
        super(KTHDataset, self).__init__()
        self.preprocess()

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
        pass



    def __len__(self):
        pass

    def __getitem__(self, index):
        pass



def main():
    KTH = KTHDataset()
    KTH.preprocess()


if __name__ == '__main__':
    main()