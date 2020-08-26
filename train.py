'''
Created on Jan 7, 2020

@author: Quang TRAN
'''

import argparse
import os
import time

import torch
import torch.nn as nn

import utils.utils as utils
import nets.gans as gans


def initParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  # @deprecated
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--crop_size", type=int, default=64, help="size of cropping area from image")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="number of batches between image sampling") 
    parser.add_argument("--test_interval", type=int, default=50, help="number of epochs between testing while training")
    parser.add_argument("--patch_size", type=int, default=3, help="the number of frames in a patch")
    parser.add_argument("--path", type=str, default="imagestry", help="output folder")
    parser.add_argument("--default_model_path", type=str, default="models_test", help="the default path of saved models")
    parser.add_argument("--adv_lambda", type=float, default=0.05, help="the default weight of adv Loss")
    parser.add_argument("--l1_lambda", type=float, default=1.0, help="the default weight of L1 Loss")
    parser.add_argument("--gdl_lambda", type=float, default=1.0, help="the default weight of GDL Loss")
    parser.add_argument("--ms_ssim_lambda", type=float, default=6.0, help="the default weight of MS_SSIM Loss")
    parser.add_argument("--isCudaUsed", type=bool, default=True, help="run with GPU or CPU (default)")
    parser.add_argument("--gen_load", type=str, default=None, help="loaded generator for training")
    parser.add_argument("--dis_load", type=str, default=None, help="loaded discriminator for training")
    
    return parser.parse_args()


def run_train(opt):
    os.makedirs("./train/" + opt.path, exist_ok=True);
    
    start = time.time()
    nets = gans.GenNet(opt)
    if opt.gen_load is not None and opt.dis_load is not None:
        generator, discriminator = nets.trainFromTrainedModel(opt.gen_load, opt.dis_load);
    else:
        generator, discriminator = nets.trainGAN()
    end = time.time()
    print('Training takes ' + str(end - start) + 's')
    
    return generator, discriminator;


def main():
    opt = initParameters()
    run_train(opt)


main()
