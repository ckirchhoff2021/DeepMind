import os
import json
import torch

import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix

from utils import *
from models.mobilenet import *


def cpu_loading_net(net, model_file):
    '''
    run model in cpu while parallel training
    '''
    state = torch.load(model_file, map_location='cpu')
    acc = state['acc']
    print('train acc: ', acc)
    state_dict = OrderedDict()
    for k, v in state['net'].items():
        name = k[7:]
        state_dict[name] = v
    net.load_state_dict(state_dict)
    net.eval()
    return net


def gpu_loading_net(net, model_file):
    '''
     run model in GPU while parallel training
    '''
    net = nn.DataParallel(net)
    state = torch.load(model_file)
    acc = state['acc']
    print('train acc: ', acc)
    net.load_state_dict(state['net'])
    net.eval()
    net.cuda()
    return net


class EmbeddingExtractor(nn.Module):
    '''
    get the special layer feature of the net
    '''
    def __init__(self, net):
        super(EmbeddingExtractor, self).__init__()
        modules = list(net.children())
        features = modules[:-1]
        classify = modules[-1]
        self.backbone = nn.Sequential(*features)
        self.fc = classify[0]

    def forward(self, image_tensor):
        vec = self.backbone(image_tensor)
        vec = vec.view(vec.size(0), -1)
        vec = self.fc(vec)
        return vec


class RecognitionInstance:
    '''
    inference instance
    '''
    def __init__(self, net, net_file, str_labels):
        self.str_labels = str_labels
        # self.net = gpu_loading_net(net, net_file)
        self.net = cpu_loading_net(net, net_file)
        self.extractor = EmbeddingExtractor(self.net)

    def predict(self, image_file):
        image_data = Image.open(image_file).convert('RGB')
        image_tensor = common_transform(image_data).unsqueeze(0)
        y_out = self.net(image_tensor)
        probs = F.softmax(y_out, dim=1)
        y_pred = y_out.max(1)[1].item()
        str_pred = self.str_labels[y_pred]
        y_prob = probs[0, y_pred].item()
        return str_pred, y_prob

    def batch_predict(self, image_folder, output_folder):
        image_list = os.listdir(image_folder)
        random.shuffle(image_list)
        for image in tqdm(image_list):
            if not image.endswith('.jpg') and not image.endswith('.png'):
                continue
            image_file = os.path.join(image_folder, image)
            try:
                pred = self.predict(image_file)
            except:
                continue

            str_pred = pred[0]
            prob = pred[1]

            if prob < 0.8:
                continue

            folder = os.path.join(output_folder, str_pred)
            if not os.path.exists(folder):
                os.mkdir(folder)

            dst = os.path.join(folder, image)
            shutil.copyfile(image_file, dst)

    def evaluate(self, test_samples):
        dataset = DatasetInstance(test_samples, train=False)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        y_preds = list()
        y_gts = list()
        correct = 0
        total = 0
        for index, (data, label) in enumerate(data_loader):
            if cuda:
                inputs, targets = data.cuda(), label.cuda()
            else:
                inputs, targets = data, label
            outputs = self.net(inputs)
            y_pred = outputs.max(1)[1]
            correct += y_pred.eq(targets).sum().item()
            total += label.size(0)
            y_preds.extend(y_pred.cpu().numpy())
            y_gts.extend(label.numpy())

        accuracy = correct / total
        confusion = confusion_matrix(y_gts, y_preds)
        print('ACC = ', accuracy)
        print(confusion)

    def predict_top3(self, image_file):
        image_data = Image.open(image_file).convert('RGB')
        image_tensor = common_transform(image_data).unsqueeze(0)
        y_out = self.net(image_tensor)
        probs = F.softmax(y_out, dim=1)
        probs = probs.squeeze()
        preds = y_out.topk(3)[1]
        preds = preds.squeeze()
        y_prob = probs[preds]

        k0 = self.str_labels[preds[0].item()]
        k1 = self.str_labels[preds[1].item()]
        k2 = self.str_labels[preds[2].item()]
        ret = dict()
        ret[k0] = y_prob[0].item()
        ret[k1] = y_prob[1].item()
        ret[k2] = y_prob[2].item()
        return ret

    def extract_feature(self, image_file):
        image_data = Image.open(image_file).convert('RGB')
        image_tensor = common_transform(image_data).unsqueeze(0)
        vec = self.extractor(image_tensor)
        return vec[0].data.numpy()


if __name__ == '__main__':
    model_file = "/Users/chenxiang/Downloads/Gitlab/Deepmind/recognition/pths/corners-mobilenetv2.pth"
    image_file = '/Users/chenxiang/Downloads/dataset/datas/sofa/corners/left/a18d518e-b00e-4a6e-8d49-791eb3ba692f.jpg'
    net = mobileV2(out_num=3)
    str_labels = ['left', 'one', 'right']
    cls = RecognitionInstance(net, model_file, str_labels)
    pred, prob = cls.predict(image_file)
    print(pred, prob)
    vec = cls.extract_feature(image_file)
    print(vec.shape)


