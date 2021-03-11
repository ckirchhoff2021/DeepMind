import pdb
import sys
import numpy as np
import torch
import os
import gc
import torch.nn as nn

from .yolo.darknet import Darknet
from .yolo.utils import get_all_boxes
from .yolo.image import correct_yolo_boxes
from utils import to_numpy, xywh_to_xyxy, keep

from torchvision.ops import nms
from torchvision.ops import boxes as box_ops

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_loss
# from mscv.cnn import normal_init

import misc_utils as utils
import ipdb

conf_thresh = 0.005
cls_thresh = 0.01
nms_thresh = 0.45
DO_FAST_EVAL = False  # 只保留概率最高的类，能够加快eval速度但会降低精度

class Model(BaseModel):
    def __init__(self, opt, logger=None):
        super(Model, self).__init__()
        self.opt = opt
        self.logger = logger
        
        # 根据YoloV2和YoloV3使用不同的配置文件
        if opt.model == 'Yolo2':
            cfgfile = 'configs/yolo2-voc.cfg'
        elif opt.model == 'Yolo3':
            cfgfile = 'configs/yolo3-coco.cfg'

        # 初始化detector
        self.detector = Darknet(cfgfile, device=opt.device).to(opt.device)
        print_network(self.detector, logger=logger)

        # 在--load之前加载weights文件(可选)
        if opt.weights:
            utils.color_print('Load Yolo weights from %s.' % opt.weights, 3)
            self.detector.load_weights(opt.weights)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, sample, *args):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        loss_layers = self.detector.loss_layers
        org_loss = []

        image = sample['image'].to(opt.device)  # target domain
        target = sample['yolo_boxes'].to(opt.device)

        detection_output = self.detector(image)  # 在src domain上训练检测

        for i, l in enumerate(loss_layers):
            ol=l(detection_output[i]['x'], target)
            org_loss.append(ol)

        loss = sum(org_loss)

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.detector.parameters(), 10000)
        self.optimizer.step()

        org_loss.clear()

        gc.collect()
        return {}

    def forward_test(self, image):
        """
        给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用
        Args:
            image: [b, 3, h, w] Tensor
        """
        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        if self.detector.net_name() == 'region':  # region_layer
            shape = (0, 0)
        else:
            shape = (opt.width, opt.height)

        num_classes = self.detector.num_classes

        outputs = self.detector(image)

        outputs = get_all_boxes(outputs, shape, conf_thresh, num_classes,
                                  device=opt.device, only_objectness=False,
                                  validation=True)

        # ipdb.set_trace()

        for b in range(image.shape[0]):
            preds = outputs[b]
            preds = preds[preds[:, 4] > conf_thresh]

            boxes = preds[:, :4]
            det_conf = preds[:, 4]
            cls_conf = preds[:, 5:]
            _, labels = torch.max(cls_conf, 1)

            b, c, h, w = image.shape

            boxes = xywh_to_xyxy(boxes, w, h)  # yolo的xywh转成输出的xyxy

            nms_indices = box_ops.batched_nms(boxes, det_conf, labels, nms_thresh)
            # nms_indices = nms(boxes, det_conf, nms_thresh)

            if len(nms_indices) == 0:
                batch_bboxes.append(np.array([[]], np.float32))
                batch_labels.append(np.array([], np.int32))
                batch_scores.append(np.array([], np.float32))
                continue

            boxes, det_conf, cls_conf = keep(nms_indices, [boxes, det_conf, cls_conf])
            max_conf, labels = torch.max(cls_conf, 1)

            scores = det_conf * max_conf

            if not DO_FAST_EVAL:  # 只保留概率最高的类，能够加快eval速度但会降低精度
                repeat_boxes = boxes.repeat([num_classes, 1])
                repeat_det_conf = det_conf.repeat(num_classes)
                repeat_conf = cls_conf.transpose(1, 0).contiguous().view(-1)
                repeat_labels = torch.linspace(0, num_classes-1, num_classes).repeat(boxes.shape[0], 1).transpose(1, 0).contiguous().view(-1)

                boxes, det_conf, cls_conf, labels = keep(repeat_conf>cls_thresh, [repeat_boxes, repeat_det_conf, repeat_conf, repeat_labels])
                scores = det_conf * cls_conf

            batch_bboxes.append(to_numpy(boxes))
            batch_labels.append(to_numpy(labels, np.int32))
            batch_scores.append(to_numpy(scores))

        return batch_bboxes, batch_labels, batch_scores


    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)



