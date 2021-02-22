import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def single_PSNR(x_clean, x_noise):
    '''
    :param x_clean:  clean image， tensor
    :param x_noise:  noise image, tensor
    :return: float
    '''
    I_max = torch.max(x_clean)
    I_max = I_max.item()
    mse = F.mse_loss(x_clean, x_noise)
    mse = mse.item()
    psnr = 10 * math.log10(I_max ** 2 / mse)
    return psnr

    
def multi_PSNR(x1, x2):
    '''
    :param x1:  clean image array
    :param x2:  noise image array
    :return: float
    '''
    
    counts = x1.size(0)
    score = 0
    for i in range(counts):
        x_clean, x_noise = x1[i], x2[i]
        psnr = single_PSNR(x_clean, x_noise)
        score += psnr
    return score / counts
    

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    x1 = torch.randn(3,3,128,128)
    x2 = torch.randn(3,3,128,128)
    psnr = multi_PSNR(x1, x2)
    print('PSNR: ', psnr)
        
        
    