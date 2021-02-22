import os
import math
import torch

import torch.nn as nn
from sr_data import *
import torch.nn.functional as F
import torch.optim as opt
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from models.srgan import Generator, Discriminator
from models.loss import PerceptualLoss, TVLoss
from indicator import *



class SRGANInstance:
    def __init__(self, train_datas, test_folder):
        self.D = Discriminator(num_channel=3)
        self.G = Generator(15, 4, num_channel=3)
        self.lr = 0.01
        self.epochs = 30
        self.batch_size = 64
        self.train_datas = train_datas
        self.test_folder = test_folder
        
        self.crop = transforms.RandomCrop((256,256))
        self.resize = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ]
        )

        
    def start_train(self):
        print('==> start training loop...')
        print('==> train num: ', len(self.train_datas))
        print('==> test num: ', len(self.test_datas))
        
        d_criterion = nn.BCELoss()
        pixel_criterion = nn.MSELoss()
        perceptural_criterion = PerceptualLoss()
        tv_criterion = TVLoss(tv_loss_weight=2e-8)
      
        d_optimizer = opt.Adam(self.D.parameters(), lr=self.lr, betas=(0.9,0.999))
        g_optimizer = opt.SGD(self.G.parameters(), lr = self.lr/100, momentum=0.9)
        
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
                hr_fake = self.G(lr_real)
                fake_out = self.D(hr_fake)
                
                perceptural_loss = perceptural_criterion(hr_fake, hr_real)
                pixel_loss = pixel_criterion(hr_fake, hr_real)
                adversarial_loss = torch.mean(1 - fake_out)
                tv_loss = tv_criterion(fake_out)
                
                g_loss = pixel_loss + adversarial_loss * 1e-3 + 6e-3 * perceptural_loss + tv_loss
                g_losses += g_loss.item()
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                
                hr_fake = self.G(lr_real)
                fake_out = self.D(hr_fake)
                fake_loss = d_criterion(fake_out, fake_label)
                
                real_out = self.D(hr_real)
                real_loss = d_criterion(real_out, real_label)
                
                d_loss = fake_loss + real_loss
                d_losses += d_loss.item()
                
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            
                if index % 10 == 0:
                    print('- Epoch: [%d]/[%d]-[%d]/[%d], d_loss = %f, g_loss = %f' %
                          (epoch, self.epochs, index, batches, d_loss.item(), g_loss.item()))
                    
            avr_loss_d = d_losses / len(train_loader)
            avr_loss_g = g_losses / len(train_loader)
            
            print('- Training: [%d]/[%d], avr_loss_d = %f, avr_loss_g = %f' % (epoch, self.epochs, avr_loss_d, avr_loss_g))
            
            with torch.no_grad():
                self.D.eval()
                self.G.eval()
                
                psnrs = 0.0
                ssims = 0.0
                cnt = 0
                test_images = os.listdir(self.test_folder)
                
                for image in test_images:
                    if not image.endswith('.png') and not image.endswith('.jpg'):
                        continue
                    image_file = os.path.join(self.test_folder, image)
                    image_data = Image.open(image_file)
                    image_crop = self.crop(image_data)
                    image_resize = self.resize(image_crop)
                    image_tensor = image_resize.unsqueeze(0)
                    y_out = self.G(image_tensor)
                    y_image = transforms.ToPILImage()(y_out)

                    cnt += 1
                    psnr = calculate_psnr(image_crop, y_image)
                    psnrs += psnr
                    ssim = calculate_ssim(image_crop, y_image)
                    ssims += ssim
                    
                psnr = psnrs / cnt
                ssim = ssims / cnt
                print('- Testing : PSNR = %f dB, SSIM = %f' % (psnr, ssim))
                
                state = {
                    'D': self.D.state_dict(),
                    'G': self.G.state_dict()
                }
                
                torch.save(state, 'output/srgan.pth')


def main():
    train_datas = get_train_set(2, '../datas/BSDS300/images/train/')
    test_datas = get_train_set(2, '../datas/BSDS300/images/test/')
    srgan = SRGANInstance(train_datas, test_datas)
    srgan.start_train()

    
if __name__ == '__main__':
    main()
