import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

import deepspeed
from deepspeed.accelerator import get_accelerator
import argparse
import json


class DeepSpeedTrainInstance(object):
    def __init__(self, train_set, test_set, train_batch_size=128, test_batch_size=64, epochs=30,
                 lr=0.005, summary_folder='./summary', output_folder='output', device_ids=[0], log_file='train.log'):
        self.lr = lr
        self.epochs = epochs
        self.train_set = train_set
        self.test_set = test_set
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device_ids = device_ids
        self.summary_folder = summary_folder
        self.output_folder = output_folder
        self.config = json.load(open('framework/ds_config.json', 'r'))
        self.logger_initialize(log_file)

    def logger_initialize(self, log_file):
        self.logger = logging.getLogger('TrainInstance')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("Finish initializing logger.")

    def run(self, model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_engine, optimizer, train_loader, __ = deepspeed.initialize(args=None,
            model=model, model_parameters=parameters, training_data=self.train_set, config=self.config)
        local_device = get_accelerator().device_name(model_engine.local_rank)
        local_rank = model_engine.local_rank

        # For float32, target_dtype will be None so no datatype conversion needed
        target_dtype = None
        if model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        elif model_engine.fp16_enabled():
            target_dtype = torch.half

        self.logger.info('==> train data num: %d', len(self.train_set))
        self.logger.info('==> test data num: %d', len(self.test_set))
        self.logger.info(f'==> device ids: {self.device_ids}')

        test_loader = DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=True, num_workers=4)
        best_acc = 0.0
        batches = len(train_loader)
        criterion = nn.CrossEntropyLoss()
        summary = SummaryWriter(self.summary_folder, comment='metric')

        for epoch in range(self.epochs):
            metric = {'corrects': 0, 'counts': 0, 'losses': 0.0}
            for index, (data, label) in enumerate(train_loader):
                inputs, targets = data.to(local_device), label.to(local_device)
                if target_dtype != None:
                    inputs = inputs.to(target_dtype)
                outputs = model_engine(inputs)
                loss = criterion(outputs, targets)

                predicts = outputs.max(1)[1]
                corrects = predicts.eq(targets).sum().item()
                acc = corrects / data.size(0)
                steps = index + epoch * batches

                summary.add_scalar('train/batch_acc', acc, steps)
                summary.add_scalar('train/batch_loss', loss.item(), steps)

                metric['counts'] += data.size(0)
                metric['corrects'] += corrects
                metric['losses'] += loss.item()

                if index % 20 == 0:
                    self.logger.info('==> Epoch:[%d]/[%d]-[%d]/[%d], training: batch_loss = %f, batch_acc = %f' %
                          (epoch, self.epochs, index, batches, loss.item(), acc))

                model_engine.backward(loss)
                model_engine.step()

            epoch_loss = metric['losses'] / len(train_loader)
            epoch_acc = metric['corrects'] / metric['counts']
            summary.add_scalar('train/epoch_acc', epoch_acc, epoch)
            summary.add_scalar('train/epoch_loss', epoch_loss, epoch)

            self.logger.info('***Training*** Epoch: [%d]/[%d]:  epoch_loss = %f, epoch_acc = %f' %
                  (epoch, self.epochs, epoch_loss, epoch_acc))

            model_engine.save_checkpoint(self.output_folder, epoch)

            model.eval()
            metric = {'corrects': 0, 'counts': 0, 'losses': 0.0}
            for index, (data, label) in enumerate(test_loader):
                inputs, targets = data.cuda(), label.cuda()
                if target_dtype != None:
                    inputs = inputs.to(target_dtype)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                predicts = outputs.max(1)[1]
                corrects = predicts.eq(targets).sum().item()
                acc = corrects / data.size(0)
                steps = index + epoch * batches

                summary.add_scalar('test/batch_acc', acc, steps)
                summary.add_scalar('test/batch_loss', loss.item(), steps)

                metric['counts'] += data.size(0)
                metric['corrects'] += corrects
                metric['losses'] += loss.item()

            epoch_loss = metric['losses'] / len(test_loader)
            epoch_acc = metric['corrects'] / metric['counts']
            summary.add_scalar('test/epoch_acc', epoch_acc, epoch)
            summary.add_scalar('test/epoch_loss', epoch_loss, epoch)
            self.logger.info('***Testing*** Epoch: [%d]/[%d]: epoch loss = %f, epoch_acc = %f' %
                  (epoch, self.epochs, epoch_loss, epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                    'acc': best_acc
                }
                torch.save(state, os.path.join(self.output_folder, 'best.pt'))
        self.logger.info("End training loop.")

    def evaluate(self, model):
        state_dict = torch.load(self.checkpoints)
        state = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            state[name] = v
        net = model
        net.load_state_dict(state)
        net.cuda()
        net.eval()
        test_loader = DataLoader(self.test_set, batch_size=64, shuffle=True, num_workers=4)
        counts = 0
        correct_num = 0
        for index, (data, label) in enumerate(test_loader):
            inputs, targets = data.cuda(), label.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            predicts = outputs.max(1)[1]
            corrects = predicts.eq(targets).sum().item()
            counts += targets.size(0)
            correct_num += corrects
        acc = correct_num / counts
        print('==> Evaluate acc: ', acc)
