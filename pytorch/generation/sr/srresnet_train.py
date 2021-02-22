import os
import math
import torch

import torch.nn as nn
from sr_data import *
import torch.optim as opt
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import save_image

from models.srresnet import SRResnet
from models.vgg_loss import PerceptualLoss
from indicator import *



class SRResnetInstance:
    def __init__(self, train_datas, test_datas):
        self.net = SRResnet(16, 4)
        self.lr = 2e-4
        self.epochs = 30
        self.batch_size = 4
        self.train_datas = train_datas
        self.test_datas = test_datas
        
        self.to_image = transforms.ToPILImage()
    
    def start_train(self):
        print('==> start training loop...')
        print('==> train num: ', len(self.train_datas))
        print('==> test num: ', len(self.test_datas))
       
        pixel_criterion = nn.MSELoss()
        perceptural_criterion = PerceptualLoss()
        optimizer = opt.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.99))
        
        train_loader = DataLoader(self.train_datas, shuffle=True, batch_size=self.batch_size, num_workers=4)
        test_loader = DataLoader(self.test_datas, shuffle=True, batch_size=1)
        batches = int(len(self.train_datas) / self.batch_size)
        
        for epoch in range(self.epochs):
            self.net.train()
            losses = 0.0
            psnrs = 0.0
            for index, (data, label) in enumerate(train_loader):
                mini_batch = data.size(0)
                inputs, targets = data, label
                outputs = self.net(inputs)
                
                perceptural_loss = perceptural_criterion(outputs, targets)
                pixel_loss = pixel_criterion(outputs, targets)
             
                loss = perceptural_loss + 0.1 * pixel_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                psnr = multi_PSNR(targets, outputs)
                psnrs += psnr
                
                if index % 10 == 0:
                    print('- Epoch: [%d]/[%d]-[%d]/[%d], percerptural_loss = %.3f, mse_loss = %.3f, loss = %.3f, psnr=%.3f(dB)' %
                          (epoch, self.epochs, index, batches, perceptural_loss.item(), pixel_loss.item(), loss.item(), psnr))
                
                if index % 20 == 0:
                    image_tensor = torch.cat((outputs, targets), 2)
                    save_image(image_tensor, 'output/srresnet/'+ str(epoch) + '-'+ str(index) + '.jpg')
            
            average_loss = losses / len(train_loader)
            average_psnr = psnrs / len(train_loader)
            print('==> Training AVR: [%d]/[%d], average_loss = %.3f, average_psnr = %/3f' % (epoch, self.epochs, average_loss, average_psnr))
            
            with torch.no_grad():
                self.net.eval()
                psnrs = 0.0
                ssims = 0.0
                for index, (data, label) in enumerate(test_loader):
                    mini_batch = data.size(0)
                    inputs, targets = data, label
                    outputs = self.net(inputs)
                    out_image = self.to_image(outputs[0])
                    psnr_value = calculate_psnr(out_image, targets[0])
                    ssim_value = calculate_ssim(out_image, targets[0])
                    psnrs += psnr
                    ssims += ssim_value
                average_psnr = psnrs / len(test_loader)
                average_ssim = ssims / len(test_loader)
                print('==> Testing AVR PSNR: %.3f(dB), AVR SSIM: %.3f' % (average_psnr, average_ssim))
                torch.save(self.net.state_dict(), 'output/srresnet.pth')


def main():
    train_datas = get_train_set(2, '../datas/BSDS300/images/train/')
    test_datas = get_train_set(2, '../datas/BSDS300/images/test/')
    srresnet = SRResnetInstance(train_datas, test_datas)
    srresnet.start_train()


if __name__ == '__main__':
    main()
