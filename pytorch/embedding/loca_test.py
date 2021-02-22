import json
import time
import torch
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn

from shapes import *
from collections import OrderedDict


def load_EmbeddingNet():
    net = EmbeddingNet()
    state = torch.load('res18.pth', map_location='cpu')
    net.load_state_dict(state)
    return net


def extract_feature(image_folder):
    feature_dict = dict()
    net = load_EmbeddingNet()
    net.eval()
    image_list = os.listdir(image_folder)
    for image in tqdm(image_list):
        if not image.endswith('.jpg'):
            continue
        room_id = image.split('.')[0]
        image_file = os.path.join(image_folder, image)
        image_data = Image.open(image_file)
        image_tensor = data_transform(image_data)
        image_tensor = image_tensor.unsqueeze(0)
        vec = net(image_tensor)
        feature = vec[0].data.numpy().tolist()
        feature_dict[room_id] = feature
    return feature_dict


def get_feature_tensor():
    features = json.load(open('config/database.json', 'r'))
    labels = list()
    values = list()
    for ikey in features.keys():
        labels.append(ikey)
        values.append(features[ikey])

    feature_tensor = torch.tensor(values)
    return feature_tensor.T, labels


def retrieval_test():
    save_path = '/Users/chenxiang/Desktop/rooms/test_result'
    test_path = '/Users/chenxiang/Desktop/rooms/test_images'
    database_path = '/Users/chenxiang/Downloads/dataset/datas/rooms'
    net = load_EmbeddingNet()
    net.eval()
    feature_tensor, feature_labels = get_feature_tensor()
    print(feature_tensor.size())

    test_list = os.listdir(test_path)
    time_consuming = 0.0
    cnt = 0

    for image in test_list:
        if not image.endswith('.jpg'):
            continue
        src = os.path.join(test_path, image)
        image_data = Image.open(src).resize((256,256))

        cnt += 1
        start = time.time()
        image_tensor = data_transform(image_data)
        image_tensor = image_tensor.unsqueeze(0)
        vec = net(image_tensor)
        dot_value = torch.mm(vec, feature_tensor)
        topk = dot_value.topk(3, dim=1)
        topk = topk[1].squeeze()
        end = time.time()
        time_consuming += (end - start)
        image_merged = Image.new('RGB', (1024, 256))
        image_merged.paste(image_data, (0,0,256,256))

        for i in range(3):
            k = topk[i].item()
            label = feature_labels[k]
            target = Image.open(os.path.join(database_path, label + '.jpg')).resize((256,256))
            image_merged.paste(target, (256 + i * 256,0, 512 + i * 256, 256))
        image_merged.save(os.path.join(save_path, image))
    print('Average time: ', time_consuming / cnt)



if __name__ == '__main__':
    get_feature_tensor()
    retrieval_test()
