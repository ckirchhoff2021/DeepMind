import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

from config import *

def parse(df, tokenizer):
    tokens = tokenizer.tokenize(open(df['file'], 'r').read())
    ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    return ids


def pad_sequences(seqs, length):
    seq = []
    mask = []
    for s in seqs:
        if len(s) < length:
            seq.append(s + [0] * (length - len(s)))
            mask.append([1] * len(s) + [0] * (length - len(s)))
        elif len(s) > length:
            seq.append(s[:length])
            mask.append([1] * length)
        else:
            seq.append(s)
            mask.append([1] * length)
    return np.stack(seq, axis=0), np.stack(mask, axis=0)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(bert_root)
    df = pd.read_csv('datas/spt.csv')

    texts = []
    cats = []
    train_val = []

    for _, content in tqdm([item for item in df.iterrows()]):
        file = content['file']
        category = content['category']
        text = tokenizer.encode(open(file, 'r').read())
        texts.append(text)
        cats.append(category)
        train_val.append(True if content['type'] == 'train' else False)

    ids, mask = pad_sequences(texts, MAX_LEN)
    cats = np.array(cats, dtype=np.int32)
    train_val = np.array(train_val, dtype=np.bool)

    np.save('datas/x.npy', ids)
    np.save('datas/y.npy', cats)
    np.save('datas/mask.npy', mask)
    np.save('datas/trainval_split.npy', train_val)

