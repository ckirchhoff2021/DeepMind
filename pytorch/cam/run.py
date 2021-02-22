import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from cam.grad_cam import GradCAM
from torchvision import models

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'image_path')


def prepare_input(src):
    image = src.copy()
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def norm_image(src):
    """
    标准化图像
    :param src: [H,W,C]
    :return:
    """
    image = src.copy()
    v = np.min(image)
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    h = np.float32(heatmap) / 255
    h = h[..., ::-1]  # bgr to rgb

    # 合并heatmap到原始图像
    cam = h + np.float32(image)
    return norm_image(cam), heatmap


def run(image_path):
    src = cv2.imread(image_path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = np.float32(cv2.resize(src, (224, 224))) / 255

    input_data = prepare_input(src)
    image_dict = {}
    net = models.resnet50(True, True)
    layer_name = get_last_conv_name(net)
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(input_data)
    image_dict['cam'], image_dict['heatmap'] = gen_cam(src, mask)
    grad_cam.remove_hook_handles()

    for ikey in image_dict.keys():
        dst = image_dict[ikey]
        if ikey == 'cam':
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(SAVE_PATH, ikey + '-b.jpg'), dst)


def main():
    path = '/Users/chenxiang/Desktop/Gauss.png'
    run(path)


if __name__ == '__main__':
    main()