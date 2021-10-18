import os
import pickle
import json

import torch
import torch.nn as nn
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, BertConfig, BertForNextSentencePrediction, BertForQuestionAnswering

import argparse

'''
parser = argparse.ArgumentParser(description='...')
parser.add_argument('--save_file', type=str, default='output/text.pkl', help='save')
parser.add_argument('--sample_file', type=int, default='youku_clip_label_train.pkl', help='xx')
args = parser.parse_args()
'''

text_root = '../../datas/youku_clip_text.pkl'
data_root = '/data/baoluo.cx/datas/videos/youku_clip'


def get_bert():
    root = '/Users/chenxiang/Downloads/bert/chinese/'
    tokenizer = BertTokenizer.from_pretrained(root)
    config = BertConfig.from_pretrained(root)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(root, config=config)
    model.eval()
    return tokenizer, model


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.root = '/Users/chenxiang/Downloads/bert/chinese/'
        self.tokenizer = BertTokenizer.from_pretrained(self.root)
        config = BertConfig.from_pretrained(self.root)
        config.output_attentions = True
        config.output_hidden_states = True
        self.model = BertModel.from_pretrained(self.root, config=config)
        self.model.eval()

    def str_forward(self, x):
        indices = self.tokenizer.encode(x)
        inputs = torch.tensor([indices])
        outputs = self.model(inputs)
        return outputs[1]

    def multi_forward(self, arrs):
        outputs = list()
        for x in arrs:
            y = self.str_forward(x)
            outputs.append(y)
        y1 = torch.cat(outputs, dim=1)
        return y1

    def forward(self, x, attention_mask=None):
        y = self.model(x, attention_mask=attention_mask)
        y1 = y[1]
        print(y1.size())
        yt = torch.mean(y1, dim=0)
        return yt


class TextDataset(Dataset):
    def __init__(self, text_dict, sample_dict):
        super(TextDataset, self).__init__()
        self.text_datas = text_dict
        self.sample_datas = sample_dict
        self.ids = list()
        self.root = '/Users/chenxiang/Downloads/bert/chinese/'
        self.tokenizer = BertTokenizer.from_pretrained(self.root)
        self.initialize()

    def initialize(self):
        for key in self.sample_datas.keys():
            if key in self.text_datas.keys():
                self.ids.append(key)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_datas[id]
        label = self.sample_datas[id]
        slice_len = 512
        overlap_len = 32
        text_list = list()

        try:
            count = len(text)
            i = 0
            while i < count:
                k1 = i
                k2 = i + slice_len
                k2 = k2 if k2 < count else count
                i = k1 + slice_len - overlap_len
                text_list.append(text[k1:k2])
        except:
            print('Error happened ...', index)
            text_list.append([','] * 10 )

        seq_list = list()
        mask_list = list()
        for text in text_list:
            indices = self.tokenizer.encode(text)
            if len(indices) < slice_len:
                seq_list.append(indices + [0] * (slice_len -len(indices)))
                mask_list.append([1] * len(indices) + [0] * (slice_len - len(indices)))
            elif len(indices) > slice_len:
                seq_list.append(indices[:slice_len])
                mask_list.append([1] * slice_len)
            else:
                seq_list.append(indices)
                mask_list.append([1] * slice_len)

        inputs = torch.tensor(seq_list)
        masks = torch.tensor(mask_list)
        return inputs, masks, label, id


def get_text_tokens(sample_dataset, save_file):
    count = len(sample_dataset)
    encode_dict = dict()
    for i in range(count):
        x, m, label, id = sample_dataset[i]
        ids = x.numpy()
        masks = m.numpy()
        encode_dict[id] = {
            'id': id,
            'mask': masks,
            'label': label,
            'token': ids
        }

    with open(save_file, 'wb') as f:
        pickle.dump(encode_dict, f)
    print('Done......')


def text_embedding(token_file, save_file):
    tokens = pickle.load(open(token_file, 'rb'))
    vocab_dict = dict()
    net = Bert()

    for ikey in tqdm(tokens.keys()):
        ivalue = tokens[ikey]
        id = ivalue['id']
        mask = ivalue['mask']
        label = ivalue['label']
        token = ivalue['token']

        x1 = torch.tensor(token)
        x2 = torch.tensor(mask)
        y = net(x1, attention_mask=x2)

        vocab_dict[id] = {
            'embedding': y.numpy().tolist(),
            'label': label
        }

    with open(save_file, 'wb') as f:
        pickle.dump(vocab_dict, f)

    print('Done...')


def main():
    text_datas = pickle.load(open(text_root, 'rb'))

    train_file = os.path.join(data_root, 'youku_clip_label_train.pkl')
    train_labels = pickle.load(open(train_file, 'rb'))

    val_file = os.path.join(data_root, 'youku_clip_label_val.pkl')
    val_labels = pickle.load(open(val_file, 'rb'))

    train_dataset = TextDataset(text_datas, train_labels)
    val_dataset = TextDataset(text_datas, val_labels)

    text_embedding(text_datas, train_dataset, 'output/train_text.pkl')
    text_embedding(text_datas, val_dataset, 'output/val_text.pkl')



def entity_embedding():
    file = '/Users/chenxiang/Downloads/Gitlab/mmkgcn_beta/materials/alphav_risk_graph_t.json'
    net = Bert()
    graph_t = json.load(open(file, 'r'))

    knots = graph_t['kgids']
    edges = graph_t['edges']

    print('knots num: ', len(knots))
    print('knots: ', knots)

    graph_n = dict()
    graph_n['kgids'] = knots
    graph_n['edges'] = edges
    graph_n['vectors'] = list()

    for knot in tqdm(knots):
        embds = net.str_forward(knot)
        embds = embds.squeeze()
        vec = np.around(embds.detach().numpy(), 4)
        graph_n['vectors'].append(vec.tolist())

    print('vector num: ', len(graph_n['vectors']))
    with open('datas/kg.json', 'w') as f:
        json.dump(graph_n, f, ensure_ascii=False)

    print('Done...')



def handle():
    kg = json.load(open('datas/kg.json', 'r'))
    graph = dict()
    count = len(kg['kgids'])
    graph['kgids'] = kg['kgids']
    graph['vectors'] = kg['vectors']
    graph['edges'] = list()
    edges = kg['edges']
    print('==> origin edges: ', len(edges))

    for e in edges:
        v1, v2 = e
        if v1 < 0 or v1 >= count:
            continue

        if v2 < 0 or v2 >= count:
            continue
        graph['edges'].append(e)

    print('==> current edges: ', len(graph['edges']))
    with open('datas/kg2.json', 'w') as f:
        json.dump(graph, f, ensure_ascii=False)


    print('Done....')




if __name__ == '__main__':
    # main()
    # text_datas = pickle.load(open(text_root, 'rb'))
    # sample_file = os.path.join(data_root, args.sample_file)
    # sample_labels = pickle.load(open(args.sample_file, 'rb'))
    #
    # sample_dataset = TextDataset(text_datas, sample_labels)
    # get_text_tokens(sample_dataset, args.save_file)

    # entity_embedding()
    handle()