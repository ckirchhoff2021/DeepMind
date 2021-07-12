import os
import random
import torch
import torch.nn as nn
from glob import glob
import pandas as pd

bert_root = '/Users/chenxiang/Downloads/bert/chinese'
thucnews_root = '/Users/chenxiang/Downloads/dataset/THUCNews'
categories = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']

NUM_TRAIN = 5000
NUM_VAL = 2000
NUM_ALL = NUM_TRAIN + NUM_VAL
MAX_LEN = 400


if __name__ == '__main__':
    types = ['train'] * NUM_TRAIN + ['val'] * NUM_VAL
    df = None
    for i, cat in enumerate(categories):
        print(cat)
        files = glob(os.path.join(thucnews_root, cat, '*.txt'))
        random.shuffle(files)
        files = files[:NUM_ALL]
        if df is None:
            df = pd.DataFrame([list(item) for item in zip([i] * NUM_ALL, types, files)],
                              columns=['category', 'type', 'file'])
        else:
            df = df.append(pd.DataFrame([list(item) for item in zip([i] * NUM_ALL, types, files)],
                              columns=['category', 'type', 'file']))

    df.to_csv('datas/spt.csv', index=0)



