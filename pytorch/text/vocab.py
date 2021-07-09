import os
import pickle

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, BertConfig, BertForNextSentencePrediction, BertForQuestionAnswering


text_root = '../../datas/youku_clip_text.pkl'


def main():
    text_datas = pickle.load(open(text_root, 'rb'))
    print(type(text_datas))
    print(len(text_datas))
    for key in text_datas:
        video_id = key
        text = text_datas[key]
        print(video_id)
        print(text)
        print(len(text))
        print(text[:256])
        break


def text_embedding():
    vocab_dict = dict()
    root = '/Users/chenxiang/Downloads/bert/chinese/'
    tokenizer = BertTokenizer.from_pretrained(root)
    config = BertConfig.from_pretrained(root)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(root, config=config)
    model.eval()

    text_datas = pickle.load(open(text_root, 'rb'))
    print(len(text_datas))

    crop_length = 256
    half_length = 128
    for key in tqdm(text_datas):
        video_id = key
        text = text_datas[key]
        count = len(text)
        text_list = list()
        i = 0
        while i < count:
            k1 = i
            k2 = i + crop_length
            k2 = k2 if k2 < count else count
            i = i + half_length
            text_list.append(text[k1:k2])

        vec = torch.zeros([1, 768])
        for value in text_list:
            encodes = tokenizer.encode(value)
            token_tensor = torch.tensor([encodes])
            outputs = model(token_tensor)
            embeddings = outputs[1]
            vec += embeddings

        vec /= len(text_list)
        vocab_dict[video_id] = vec.data.numpy().tolist()

    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f, 'ensure_ascii=False')

    print('Done...')



if __name__ == '__main__':
    # main()
    text_embedding()