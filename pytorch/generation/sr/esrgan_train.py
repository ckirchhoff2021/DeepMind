import os
import math
import torch

import torch.nn as nn
from sr.sr_data import *
import torch.nn.functional as F
import torch.optim as opt
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms, models

from sr_data import *
from indicator import *
from models.vgg_loss import PerceptualLoss
from models.esrgan import ESRGANDiscriminator, RRDBNet



class ESRGANInstance:
    def __init__(self, train_datas, test_datas):
        self.D = ESRGANDiscriminator(3)
        self.G = RRDBNet(3, 3, 64, 23, gc=32)
        self.lr = 0.01
        self.epochs = 100
        self.batch_size = 64
        
        self.train_datas = train_datas
        self.test_datas = test_datas
    
    def start_train(self):
        print('==> start training loop...')
        print('==> train num: ', len(self.train_datas))
        print('==> test num: ', len(self.test_datas))
        
        pixel_criterion = nn.MSELoss()
        adversarial_criterion = nn.BCELoss()
        perceptual_criterion = PerceptualLoss()
        
        d_optimizer = opt.Adam(self.D.parameters(), lr=self.lr, betas=(0.9, 0.999))
        g_optimizer = opt.SGD(self.G.parameters(), lr=self.lr / 100, momentum=0.9)
    
        train_loader = DataLoader(self.train_datas, shuffle=True, batch_size=self.batch_size, num_workers=4)
        test_loader = DataLoader(self.test_datas, shuffle=True, batch_size=self.batch_size, num_workers=4)
        
        batches = int(len(self.train_datas) / self.batch_size)
        for epoch in range(self.epochs):
            self.D.train()
            self.G.train()
            d_losses = 0.0
            g_losses = 0.0
            for index, (data, label) in enumerate(train_loader):
                mini_batch = data.size(0)
                real_label = torch.ones(mini_batch, 1)
                fake_label = torch.zeros(mini_batch, 1)
                
                lr_real, hr_real = data, label
                probs_real = self.D(hr_real)
                hr_fake = self.G(lr_real)
                probs_fake = self.D(hr_fake)
                
                probs_rf = probs_real - probs_fake.mean()
                probs_fr = probs_fake - probs_real.mean()
                
                adv_loss_rf = adversarial_criterion(probs_rf, fake_label)
                adv_loss_fr = adversarial_criterion(probs_fr, real_label)
                
                advesarial_loss = (adv_loss_fr + adv_loss_rf) / 2
                perceptual_loss = perceptual_criterion(hr_fake, hr_real)
                content_loss = pixel_criterion(hr_fake, hr_real)
                
                generator_loss = 0.1 * content_loss + 5e-3 * advesarial_loss  + 1.0 * perceptual_loss
                
                g_optimizer.zero_grad()
                generator_loss.backward()
                g_optimizer.step()
                
                probs_real = self.D(hr_real)
                hr_fake = self.G(lr_real)
                probs_fake = self.D(hr_fake)
                
                probs_rf = probs_real - probs_fake.mean()
                probs_fr = probs_fake - probs_real.mean()
              
                adv_loss_rf = adversarial_criterion(probs_rf, real_label)
                adv_loss_fr = adversarial_criterion(probs_fr, fake_label)
                discriminator_loss = (adv_loss_fr + adv_loss_rf) / 2
               
                self.d_optimizer.zero_grad()
                discriminator_loss.backward()
                self.d_optimizer.step()
                
                if index % 10 == 0:
                    print('- Epoch: [%d]/[%d]-[%d]/[%d], d_loss: %.4f, g_loss: %4f, adversarial_loss: %.4f, '
                          'perceptual_loss: %.4f, content_loss: %4f' %
                          (epoch, self.epochs, index, batches, discriminator_loss.item(), generator_loss.item(), advesarial_loss.item(),
                           perceptual_loss.item(), content_loss.item()))
                    
                    if index % 20 == 0:
                        images = torch.cat((real_high, fake_high), 2)
                        save_image(images, 'output/' + str(epoch) + '-' + str(index) + '.jpg')
                
                d_losses += discriminator_loss.item()
                g_losses += generator_loss.item()
            
            avr_loss_d = d_losses / len(train_loader)
            avr_loss_g = g_losses / len(train_loader)
            
            print('==> AVR: [%d]/[%d], avr_loss_d = %f, avr_loss_g = %f' % (epoch, self.epochs, avr_loss_d, avr_loss_g))
            
            with torch.no_grad():
                self.D.eval()
                self.G.eval()
                psnrs = 0.0
                for index, (data, label) in enumerate(test_loader):
                    mini_batch = data.size(0)
                    inputs, targets = data, label
                    fake_data = self.G(inputs)
                    mse = F.mse_loss(fake_data, targets)
                    psnr = 10 * math.log10(1 / mse.item())
                    psnrs += psnr
                avr_psnr = psnrs / len(test_loader)
                print('- AVR PSNR: %f dB' % (avr_psnr))
                
                state = {
                    'D': self.D.state_dict(),
                    'G': self.G.state_dict()
                }
                
                torch.save(state, 'output/esrgan.pth')


def main():
    train_datas = get_train_set(2, '../datas/BSDS300/images/train/')
    test_datas = get_train_set(2, '../datas/BSDS300/images/test/')
    srgan = ESRGAN(train_datas, test_datas)
    srgan.start_train()


if __name__ == '__main__':
    main()
