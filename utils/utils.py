'''
Created on Apr 10, 2020

@author: Quang TRAN
'''

import numpy as np
import math
import cv2

import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

import utils.calculator as cal


def transform_tensor_to_img(tensor):
    if tensor is None:
        return None
    
    temp = tensor if tensor.get_device() == -1 else tensor.cpu()  # copy to cpu if neccessary
    return transforms.ToPILImage()(temp).convert("RGB")


def resize_img(img, sizeX, sizeY):
    transResize = transforms.Resize((sizeX, sizeY))
    
    return transResize(img)


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size=size)


def convert_numpy_array_to_image(array):
    return Image.fromarray(array)

   
def cal_psnr_tensor(img1, img2):
    '''
    Calculate PSNR from two tensors of images
    :param img1: tensor
    :param img2: tensor
    '''
    diff = (img1 - img2)
    diff = diff ** 2
        
    if diff.sum().item() == 0:
        return float('inf')

    rmse = diff.sum().item() / (img1.shape[0] * img1.shape[1] * img1.shape[2])
    psnr = 20 * np.log10(1) - 10 * np.log10(rmse)
        
    return psnr

    
def cal_psnr_img(img1, img2):
    '''
    Calculate PSNR from two image
    :param img1: numpy array
    :param img2: numpy array
    '''
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
        
    if mse == 0:
        return float('inf')
        
    return 20 * math.log10(255.0 / math.sqrt(mse))


def get_sub_name_from_path(file_name):
    '''
    Target: D:/gan_testing/data/cuhk/train\16.avi_244.jpg
    :param file_name: file path
    '''
    start = "\\"
    end = ".avi_"
    
    if (file_name.find(end) == -1):
        return "";
    
    temp = file_name[file_name.find(start) + 1:]
    
    if temp == -1:
        return "";
    
    return temp[:temp.find(end)]


def write_to_text_file(path, content):
    '''
    create new file then write
    :param path:
    :param content:
    '''
    out = open(path, 'w')
    out.write(content)
    out.close();

    
def write_to_existed_text_file(path, content):
    '''
    continue to write into the file
    :param path:
    :param content:
    '''
    out = open(path, 'a')
    out.write(content)
    out.close();

    
def _ssim(img1, img2):
    '''
    K1 = 0.01
    K2 = 0.03
    L = 255
    :param img1: [0,255]
    :param img2: [0,255]
    '''
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * 
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cal_ssim_img(img1, img2):
    return cal.cal_ssim_img(img1, img2)


def cal_ssim_tensor(X, Y,):
    return cal.cal_ssim_tensor(X, Y, data_range=1.0)


def cal_closest_size(size, factor=16):
    '''
    find the closest size from input
    :param size:
    :param factor: no.times image's size will be reduced
    '''
    if size % factor == 0 : return size;
    
    return (int((size / factor) / 5) * 5 + 5) * factor;


def transform_feature_map_to_img(ft_map):
    '''
    
    :param ft_map: [1, n_features, size, size]
    '''
    images_per_row = 16
    n_features = ft_map.shape[1]
    size = ft_map.shape[2]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = ft_map[0, col * images_per_row + row, :, :]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(np.array(channel_image.detach().cpu()), 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,  # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
#     figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0])
    
    return display_grid
