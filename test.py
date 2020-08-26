# -*- coding: utf-8 -*-
"""
Created on Aug 20, 2020

@author: Quang TRAN
"""

from PIL import Image
import warnings
from warnings import warn
import os
import sys
import numpy as np
import glob
import random as r
import time
import math
import cv2
import cupy
import re
import functools

import torch.cuda
import torch.optim
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.autograd import Variable

dataset_path = "dataset_path"
output_path = "dir_path"

net_path = "frame_intepolation_path.pt"
ref_net_path = "refinement_path.pt"


def get_sub_name_from_path(file_name):
    '''
    :param file_name: file path
    '''
    start = "\\"
    end = ".avi_"
    
    if (file_name.find(end) == -1):
        return "";
    
    temp = file_name[file_name.rfind(start) + 1:]
    
    if temp == -1:
        return "";
    
    return temp[:temp.rfind(end)]

# Custom crop image transformation
class CropTransform:

    def __init__(self, h, w, size):
        self.h_coor = h;
        self.w_coor = w;
        self.size = size;

    def __call__(self, img):
        return TF.crop(img, self.h_coor, self.w_coor, self.size, self.size);

'''
Load dataset support
'''

def transformImage(img, h, w):
    size = min(img.size)
    transTensor = transforms.ToTensor();        
    transResize = transforms.Resize(240)
        
    return transTensor(transResize(img)).numpy();

def calculate_padding_conv(w_in, w_out, kernel_size, stride):
	'''
	w_out = (w_in-F+2P) / S + 1
	w_out: width of output
	w_in: width of input 
	'''
	return ((w_out - 1) * stride - w_in + kernel_size) // 2;

# Generator network
class GeneratorWithCondition_NoNoise_V7(nn.Module):
    '''
    A generator without noise z
    '''

    def __init__(self):
        super(GeneratorWithCondition_NoNoise_V7, self).__init__()
        self.nfg = 64  # the size of feature map
        self.c = 3  # output channel
        filter_size = 4
        stride_size = 2
        
        self.down_sample_blocks = nn.Sequential(
            nn.Conv2d(self.c * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(self.nfg * 8),
            nn.LeakyReLU(0.02, inplace=True)
            )
        
        self.up_sample_block = nn.Sequential(
            nn.ConvTranspose2d(self.nfg * 8, self.nfg * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.nfg * 4, self.nfg * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.nfg * 2, self.nfg, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(self.nfg),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.nfg, self.c, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.Tanh()
            )
    
    def forward(self, data1, data2):
      h1 = int(list(data1.size())[2])
      w1 = int(list(data1.size())[3])
      h2 = int(list(data2.size())[2])
      w2 = int(list(data2.size())[3])
      
      if h1 != h2 or w1 != w2:
        return sys.exit('Frame size problem')

      h_padded = False
      w_padded = False

      if (h1 % 32 != 0 or (h1 - w1) < 0):
        pad = 32 - (h1 % 32) if (h1 - w1) >= 0 else 32 - (h1 % 32) + (w1 - h1)
        data1 = torch.nn.functional.pad(data1, (0, 0, 0, pad))
        data2 = torch.nn.functional.pad(data2, (0, 0, 0, pad))
        h_padded = True

      if (w1 % 32 != 0 or (h1 - w1) > 0):
        pad = 32 - (w1 % 32) if (h1 - w1) <= 0 else 32 - (h1 % 32) + (h1 - w1)
        data1 = torch.nn.functional.pad(data1, (0, pad, 0, 0))
        data2 = torch.nn.functional.pad(data2, (0, pad, 0, 0))
        w_padded = True

      out = torch.cat((data1, data2), 1)  # @UndefinedVariable

      out_down = self.down_sample_blocks(out)
      out_up = self.up_sample_block(out_down)

      if (h_padded):
        out_up = out_up[:, :, 0:h1, :]
      if (w_padded):
        out_up = out_up[:, :, :, 0:w1]
          
      return out_up


# refinement network
class UnetGenerator(nn.Module):
    """Create a Unet-based self.net_G"""

    def __init__(self, input_nc, output_nc, num_downs=6, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet self.net_G
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 6,
                                image of size 64x64 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        h1 = int(list(input.size())[2])
        w1 = int(list(input.size())[3])

        h_padded = False
        w_padded = False

        if (h1 % 32 != 0 or (h1 - w1) < 0):
            pad = 32 - (h1 % 32)
            input = torch.nn.functional.pad(input, (0, 0, 0, pad))
            h_padded = True

        if (w1 % 32 != 0 or (h1 - w1) > 0):
            pad = 32 - (w1 % 32)
            input = torch.nn.functional.pad(input, (0, pad, 0, 0))
            w_padded = True

        out = self.model(input)

        if (h_padded):
            out = out[:, :, 0:h1, :]
        if (w_padded):
            out = out[:, :, :, 0:w1]

        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)  # @UndefinedVariable
	#end
	
#end

def transform_tensor_to_img(tensor):
	if tensor is None:
		return None

	temp = tensor if tensor.get_device() == -1 else tensor.cpu()  # copy to cpu if neccessary
	return transforms.ToPILImage()(temp).convert("RGB")

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

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def cal_ssim_img(img1, img2):
	'''calculate SSIM
	the same outputs as MATLAB's
	:param img1, img2: [0, 255]
	src: https://cvnote.ddlee.cn/2019/09/12/PSNR-SSIM-Python.html#numpy-implementation-1
	'''
	if not img1.shape == img2.shape:
		raise ValueError('Input images must have the same dimensions.')
	if img1.ndim == 2:
		return _ssim(img1, img2)
	elif img1.ndim == 3:
		if img1.shape[2] == 3:
			ssims = []
			for i in range(3):
				ssims.append(_ssim(img1, img2))
			return np.array(ssims).mean()
		elif img1.shape[2] == 1:
			return _ssim(np.squeeze(img1), np.squeeze(img2))
	else:
		raise ValueError('Wrong input image dimensions.')
	
def cal_metrics(ground_truth, gen_imgs):
	if (gen_imgs.data[0].shape[1] != ground_truth.data[0].shape[1]):
		warn('Different in size of output and ground truth.')

	gen_img = transform_tensor_to_img(gen_imgs.data[0])
	ground_truth_img = transform_tensor_to_img(ground_truth.data[0])

	psnr_tensor1 = cal_psnr_tensor(gen_imgs.data[0], ground_truth.data[0])
	ssim_img1 = cal_ssim_img(np.array(gen_img), np.array(ground_truth_img))

	return psnr_tensor1, ssim_img1

# Load network for testing
def load_gen_for_evaluation(model):
	'''
	Load model for evaluation
	'''
	
	path = net_path
	checkpoint = torch.load(path)
	start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	model.cuda()
	model.eval()
	return start_epoch, model


def run_with_one_sample(frame1, frame2, gt_frame, index, file_name, model, model2, ref_model, step=1):
  if ref_model is not None:
    return _run_with_one_sample_with_refinement(frame1, frame2, gt_frame, index, file_name, model, ref_model, step)
  if model2 is None:
    return _run_with_one_sample_with_one_model(frame1, frame2, gt_frame, index, file_name, model, step)

  return _run_with_one_sample_with_two_models(frame1, frame2, gt_frame, index, file_name, model, model2, step)

def _run_with_one_sample_with_one_model(frame1, frame2, gt_frame, index, file_name, model, step):
  temp_start = time.time()
  gen_imgs = model(frame1, frame2)
  gen_time = time.time() - temp_start
  psnr, ssim = cal_metrics(gt_frame, gen_imgs)

  if index % step == 0:
    temp = torch.cat(((frame1 + frame2)/2, gt_frame, gen_imgs))

  return psnr, ssim, gen_time, gen_imgs, None
	
def _run_with_one_sample_with_refinement(frame1, frame2, gt_frame, index, file_name, gen_net, ref_net, step):
	temp_start = time.time()
	gen_imgs = gen_net(frame1, frame2)
	ref_imgs = ref_net(gen_imgs)
	gen_time = time.time() - temp_start
	psnr, ssim = cal_metrics(gt_frame, ref_imgs)

	return psnr, ssim, gen_time, gen_imgs, ref_imgs

def run_with_load(path, generator, predictor=None, refinement=None, n_epoch=-1):
	if (not os.path.isdir(path)):
		raise FileNotFoundError("ImageDatasetLoader: Cannot locate %s" % (path))
	gen_name = "gennet_gen_images53"
	out_path = output_path
	os.makedirs(out_path, exist_ok=True)

	print("Start to test model %s at %d epochth..." % (generator, n_epoch))
	if refinement is not None: print("Start to test model %s..." % (refinement))
	log = "Start to test model %s at %d epochth..." % (generator, n_epoch)
	log += "\nFile\tPSNR\tSSIM"
	print("Testing....")
	print("output: " + out_path)

	imgs = []
	h = 0
	w = 0
	count = 0
	psnr_list = []
	ssim_list = []
	time_list = []
	current_progress = 0

	files = glob.glob(path + '/*.jpg')
	files.extend(glob.glob(path + "/*.png"))
	files.sort()
	print("Loaded %d frames." % len(files))
 
	file_name1 = None
	file_name2 = None

	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	for file_name in files:
		img = Image.open(file_name)
		imgs.append(img.copy())
		img.load()
		img.close()
		count+=1
    
		process = 100.0 * count / len(files)
		if int(process) - current_progress >= 10:
			print(str(round(process)) + "%")
			current_progress = int(process)
   
		temp_name = get_sub_name_from_path(file_name)

		if len(imgs) == 1:
			file_name1 = temp_name
		elif len(imgs) == 2 and file_name1 == temp_name:
			file_name2 = temp_name
		elif len(imgs) == 2:
			del imgs[0]
			file_name1 = temp_name
			file_name2 = None
		elif len(imgs) > 2 and file_name2 != temp_name:
			imgs.clear()
			file_name1 = None
			file_name2 = None
		else:
			# process data
			frame1 = Variable(Tensor(transformImage(imgs[0], h, w)));
			gt_frame = Variable(Tensor(transformImage(imgs[1], h, w)));
			frame2 = Variable(Tensor(transformImage(imgs[2], h, w)));
			frame1 = frame1.view(1, frame1.shape[0], frame1.shape[1], frame1.shape[2])
			gt_frame = gt_frame.view(1, gt_frame.shape[0], gt_frame.shape[1], gt_frame.shape[2])
			frame2 = frame2.view(1, frame2.shape[0], frame2.shape[1], frame2.shape[2])
			
			out = torch.rand(gt_frame.shape)
			ref = torch.rand(gt_frame.shape)
			temp_ref = None
			psnr=[]
			ssim=[]
			time=[]
			for i in range(round(gt_frame.shape[2]/64)):
				for j in range(round(gt_frame.shape[3]/64)):
					dx = i * 64
					dy = j * 64
					temp1 = frame1[:, :, dx:dx+64, dy:dy+64].clone()
					temp2 = frame1[:, :, dx:dx+64, dy:dy+64].clone()
					temp_gt = gt_frame[:, :, dx:dx+64, dy:dy+64].clone()
					temp_psnr, temp_ssim, temp_time, temp_out, temp_ref = run_with_one_sample(temp1, temp2, temp_gt, count, temp_name, generator, predictor, refinement)
					psnr.append(temp_psnr)
					ssim.append(temp_ssim)
					time.append(temp_time)
					out[:, :, dx:dx+64, dy:dy+64] = temp_out.clone()[:,:,:temp1.shape[2],:temp1.shape[3]]
					if temp_ref is not None: ref[:, :, dx:dx+64, dy:dy+64] = temp_ref.clone()[:,:,:temp1.shape[2],:temp1.shape[3]]
			# update tracker
			if True:
				temp = torch.cat(((frame1 + frame2)/2, gt_frame, out.cuda())) if temp_ref is None else torch.cat((gt_frame, out.cuda(), ref.cuda()))
				save_image(temp, "%s/all_%s_%d.png" % (output_path, temp_name, count), nrow=3, padding=10)
			del imgs[0]
			file_name2 = temp_name
			file_name1 = file_name2
			# generate data
			log += "\n%s_%d\t%f\t%f" % (temp_name, count, np.average(np.array(psnr)), np.average(np.array(ssim)))
			psnr_list.append(np.average(np.array(psnr)))
			ssim_list.append(np.average(np.array(ssim)))
			time_list.append(np.average(np.array(time)))

	minPSNR = min(psnr_list)
	maxPSNR = max(psnr_list)
	avgPSNR = np.average(np.array(psnr_list))
	minSSIM = min(ssim_list)
	maxSSIM = max(ssim_list)
	avgSSIM = np.average(np.array(ssim_list))
	avgTime = np.average(np.array(time_list))

	print("Test on %d patches." % (len(files)))
	print("Min/Max/Avg PSNR value of %s is %f/%f/%f dB" % (gen_name, minPSNR, maxPSNR, avgPSNR))
	print("Min/Max/Avg SSIM value of %s is %f/%f/%f dB" % (gen_name, minSSIM, maxSSIM, avgSSIM))
	print("Average generate time: %f s." % (avgTime))
	print("Done.")

	out = open(out_path + "/log.txt", 'w')
	out.write(log)
	out.close();
   
	return;

def test_with_my_proposed():
  path = net_path
  checkpoint = torch.load(path)
  epoch = checkpoint['epoch']
  model = GeneratorWithCondition_NoNoise_V7()
  model.load_state_dict(checkpoint['state_dict'])
  model.cuda()
  model.eval()
  print('Model loaded.')
  run_with_load(path=dataset_path, generator=model, n_epoch=epoch)

#end

def test_with_my_proposed_with_refinement():
	gen_path = net_path
	ref_path = ref_net_path
	
	# load generator
	ckp_gen = torch.load(gen_path)
	epoch = ckp_gen['epoch']
	gen_net = GeneratorWithCondition_NoNoise_V7()
	gen_net.load_state_dict(ckp_gen['state_dict'])
	gen_net = gen_net.cuda()
	gen_net.eval()
	print('Generator loaded.')
	
	# load refinement
	ckp_ref = torch.load(ref_path)
	ref_net = UnetGenerator(3, 3)
	ref_net.load_state_dict(ckp_ref['state_dict'])
	ref_net = ref_net.cuda()
	ref_net.eval()
	print('Refinement network pix2pix loaded.')
	
	# run test
	run_with_load(path=dataset_path, generator=gen_net, n_epoch=epoch, refinement=ref_net)
	
	
#end

#run_test(1)
print(output_path)
os.makedirs(output_path, exist_ok=True);
test_with_my_proposed_with_refinement()