import tensorflow as tf
import tensorflow.keras as keras
import glob
import os

from PIL import Image
import numpy as np
import cv2
import random
import json



class FrameDatas(tf.keras.utils.Sequence):
    def __init__(self, sample_list, batch_size, image_size):
        self.samples = sample_list
        self.batch_size= batch_size
        self.image_size = image_size
        self.labels = ['50', '160', '320', '640']

    def __load__(self, sample):
        frames, label, _ = sample
        if label == '500':
            label = '320'

        int_label = self.labels.index(label)
        count = len(frames)
        mid = count // 2  
        q1 = frames[mid]
        q2 = frames[mid+1]
        # q3 = frames[mid+2]
        # q4 = frames[mid+3]

        i1 = Image.open(q1)
        i2 = Image.open(q2)
        # i3 = Image.open(q3)
        # i4 = Image.open(q4)

        x1 = i1.resize(self.image_size)
        x2 = i2.resize(self.image_size)

        x1 = np.array(x1).astype(np.float32) / 255.0
        x2 = np.array(x2).astype(np.float32) / 255.0

        y = np.zeros(4) 
        y[int_label] = 1
        x = np.concatenate([x1, x2], axis=-1)
        return x, y

    def __getitem__(self, index):
        i1 = self.batch_size * index
        i2 = self.batch_size * (index + 1)
        if i2 > len(self.samples):
            i2 = len(self.samples)

        batch_samples = self.samples[i1:i2]
        images = []
        labels = []
        for sample in batch_samples:
            image, label= self.__load__(sample)
            images.append(image)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.samples)//self.batch_size
    
    
if __name__=='__main__':
    sample_file = '../MotionClassify/datas/sample.json'
    sample_list = json.load(open(sample_file, 'r'))
    Data = FrameDatas(sample_list, 8, (384, 512))   

    x, y = Data[1]
    print(x.shape)
    print(y.shape)
