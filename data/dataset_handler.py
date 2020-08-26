'''
Created on Apr 10, 2020

@author: Quang TRAN
'''

import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

import numpy as np
import glob
import random as r
import os

from PIL import Image
from warnings import warn
from enum import Enum

import utils.utils as utils


class Dataset(Enum):
    D_UCF101_BODY_TRAIN = 'D:/gan_testing/data/ucf101_body_motion/train'
    D_UCF101_BODY_TEST = 'D:/gan_testing/data/ucf101_body_motion/test_49'


# Custom crop image transformation
class CropTransform:

    def __init__(self, h, w, size):
        self.h_coor = h;
        self.w_coor = w;
        self.size = size;

    def __call__(self, img):
        return TF.crop(img, self.h_coor, self.w_coor, self.size, self.size);


# Custom image loader including load, transform, crop and group
class ImageDatasetLoader:
    
    def __init__(self, opt):
        self.opt = opt;
        self.dataset = np.array([]);
        self.psnr_threshold = 12
        self.full_frame = self.opt.full_frame_mode
        self.crop_size = self.opt.crop_size
    
    def loadImageDatasetForNoNoise(self, path):
        if (not os.path.isdir(path)):
            raise FileNotFoundError("ImageDatasetLoader: Cannot locate %s" % (path))

        imgs = []
        dataset = [];
        h = -1
        w = -1
        files = glob.glob(path + '/*.jpg')
        files.sort()
        no_files = len(files)
        current_progress = 0
        
        pre_file_name_1 = ""
        pre_file_name_2 = ""

        for file_name in files:
            count = len(imgs)
            progress = 100.0 * count / no_files
            if (int(progress) - current_progress >= 10) :
                current_progress = int(progress)
                print(str(int(current_progress)) + "%")
            
            img = Image.open(file_name)
            imgs.append(img.copy())
            count += 1
        
            img_size = min(img.size)
            temp_name = utils.get_sub_name_from_path(file_name)
                
            if (count > 2):
                pre_file_name_1 = pre_file_name_2
                pre_file_name_2 = temp_name
                
                if (pre_file_name_1 != pre_file_name_2):
                    continue;
                h = r.randint(0, max(0, img_size - self.opt.img_size - 1))
                w = r.randint(0, max(0, img_size - self.opt.img_size - 1))
                data = []
                data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 3], h, w));
                data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 2], h, w));
                data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 1], h, w));
                dataset.append(np.asarray(data))
            elif (count == 1):
                pre_file_name_2 = temp_name
            else:
                pre_file_name_1 = temp_name
                
            img.load()
            
        # end
                
        if len(dataset) == 0:
            warn("No data loaded.")
            return dataset
        
        self.dataset = np.asarray(dataset)
        
        return self.dataset;

    
    def _transformImageForNoNoiseGenerator(self, img, h, w):
        size = min(img.size)
        transTensor = transforms.ToTensor();
        
        if (self.crop_size == -1) :
            return transTensor(img).numpy();
        else:
            transCrop = CropTransform(h, w, min(self.crop_size, size));
        
        return transTensor(transCrop(img)).numpy();
    
    def extracInput(self, data):
        '''
        From [3 * batch_size,c,w,h] in put, extract the two inputs for network and one expected output
        :param data: array [batch_size, c*2,w,h]
        :output array [batch_size, c,w,h]
        '''
        previouses = []
        laters = []
        middles = []
        
        patches = np.reshape(data, (self.opt.batch_size, self.opt.patch_size, self.opt.channels, self.opt.img_size, self.opt.img_size))
        
        for imgs in patches:
            previouses.append(np.asarray(imgs[0]))
            laters.append(np.asarray(imgs[self.opt.patch_size - 1]))
            middles.append(np.asarray(imgs[(self.opt.patch_size - 1) // 2]));
        
        middles = np.asarray(middles)
        previouses = np.asarray(previouses)
        laters = np.asarray(laters)
        
        return previouses, laters, middles;
