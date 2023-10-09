import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from enum import Enum
import numpy as np
import glob
import random as r
from PIL import Image


FILE_TEMPLATE = 'frame_{0:09d}.jpg'

class Paths(Enum):
    #TRAIN_DATA_PATH = '../data/frames/ghibli256'
    TRAIN_DATA_PATH = '../Kinetics400/frames'
    TEST_DATA_PATH = '../data/frames/ghibli/test'

# Custom crop image transformation
class CropTransform:

    def __init__(self, h, w, size):
        self.h_coor = h;
        self.w_coor = w;
        self.size = size;

    def __call__(self, img):
        return TF.crop(img, self.h_coor, self.w_coor, self.size, self.size);

class FrameDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt;
        self.path = Paths.TRAIN_DATA_PATH.value

    def __len__(self):
        files = glob.glob(self.path + '/*.jpg')
        return len(files) - 2

    def __getitem__(self, idx):
        #print(idx)
        img1 = Image.open(os.path.join(self.path, FILE_TEMPLATE.format(idx)))
        img2 = Image.open(os.path.join(self.path, FILE_TEMPLATE.format(idx+1)))
        img3 = Image.open(os.path.join(self.path, FILE_TEMPLATE.format(idx+2)))
        #img1 = img1.resize((64,64), Image.Resampling.LANCZOS)
        #img2 = img2.resize((64,64), Image.Resampling.LANCZOS)
        #img3 = img3.resize((64,64), Image.Resampling.LANCZOS)
        img_size = min(img1.size)

        h = r.randint(0, max(0, img_size - self.opt.img_size - 1))
        w = r.randint(0, max(0, img_size - self.opt.img_size - 1))
        data = []
        data.append(self._transformImageForNoNoiseGenerator(img1, h, w));
        data.append(self._transformImageForNoNoiseGenerator(img2, h, w));
        data.append(self._transformImageForNoNoiseGenerator(img3, h, w));

        #image1 = read_image(img_path1)
        return np.asarray(data)
        
    def _transformImageForNoNoiseGenerator(self, img, h, w):
        size = min(img.size)
        transTensor = transforms.ToTensor();
        
        if (self.opt.crop_size == -1) :
            return transTensor(img).numpy();
        else:
            transCrop = CropTransform(h, w, min(self.opt.crop_size, size));
        
        return transTensor(transCrop(img)).numpy();