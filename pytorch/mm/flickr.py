import json
import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader, Dataset

annotations = '/Users/chenxiang/Downloads/dataset/flickr30k/results_20130124.token'


class Flickr30(Dataset):
    def __init__(self):
        super(Flickr30, self).__init__()
        self.annotation_dict = dict()
        self.initialize()

    def initialize(self):
        fs = open(annotations, 'r')
        while 1:
            line = fs.readline()
            if not line:
                break
            k = line.index('#')
            id = line[:k]



def preprocess():
    annotation_dict = dict()
    fs = open(annotations, 'r')

    while 1:
        line = fs.readline()
        if not line:
            break
        k = line.index('#')
        id = line[:k]
        k1 = k+1

        while k1 < len(line):
            if line[k1] >= '0' and line[k1] <= '9':
                k1 += 1
            else:
                break

        label = line[k+1:k1]
        tokens = line[k1:].strip()
        if id not in annotation_dict.keys():
            annotation_dict[id] = ['0'] * 5

        annotation_dict[id][int(label)] = tokens

    with open('flickr30_annotations.json', 'w') as f:
        json.dump(annotation_dict, f, ensure_ascii=False)

    print('Done...')


if __name__ == '__main__':
    preprocess()
