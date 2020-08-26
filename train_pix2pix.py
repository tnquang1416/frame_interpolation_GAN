'''
Created on Aug 13, 2020

@author: Quang TRAN
'''
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.cuda as cuda
import functools
import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

import data.dataset_handler as dataset_handler
import utils.utils as utils
import nets.generators as generators


class UnetGenerator(nn.Module):
    """
	The orginal implementation is https://github.com/phillipi/pix2pix
	Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
	"""

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
        return self.model(input)

    def get_net_params(self):
        for name, params in self.named_parameters():
            print(name, params.size())


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


class NLayerDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator
    Require two images as input: both input and output of the generator
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc * 2, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=0)]  # output 1 channel prediction map
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    def get_net_params(self):
        for name, params in self.named_parameters():
            print(name, params.size())

    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class pix2pix():
    '''
    pix2pix class supports transfer image one by one.
    '''
    
    def __init__(self, opt):
        self.net_G = UnetGenerator(opt.channels, opt.channels)
        self.net_D = NLayerDiscriminator(opt.channels)
        self.net_interpolate = generators.GeneratorWithCondition_NoNoise_V7(opt)
        self.opt = opt
        self.data_loader = None
        self.data_path = dataset_handler.Dataset.D_UCF101_BODY_TRAIN.value
        
        # Loss function
        self.adv_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        if cuda.is_available() and opt.isCudaUsed:
            self.net_G = self.net_G.cuda()
            self.net_D = self.net_D.cuda()
            self.adv_loss = self.adv_loss.cuda()
            self.l1_loss = self.l1_loss.cuda()
        
        # Initialize weights
        self.net_G.apply(weights_init_normal)
        self.net_D.apply(weights_init_normal)
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
    def load_training_dataset(self):
        if self.data_loader is not None:
            return self.data_loader
        
        print("Loading training dataset: %s" % self.data_path)
        dataset = dataset_handler.ImageDatasetLoader(self.opt).loadImageDatasetForNoNoise(self.data_path);
        self.data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        print("Done.")
        
        return self.data_loader   

    def load_interpolation_network(self):
        path = "D:/pyProject/GANs/implementations/my_approach/models_test/cpt/gennet_gen_images54_400.pt"
        self._load_model(self.net_interpolate, path)
    
    def train(self, opt, progress=0):
        print(opt)
        print(self.net_G)
        print(self.net_D)
        print(self.net_interpolate)
        
        self.load_interpolation_network()
        
        self.load_training_dataset()
        
        if progress <= 0:
            no_epochs = self.opt.n_epochs
            progress = 0
        else:
            no_epochs = self.opt.n_epochs - progress
        
        os.makedirs(self.opt.path, exist_ok=True)
        self._write_to_file()
        
        # ----------
        #  Training
        # ----------    
        for t_epoch in range(no_epochs):
            epoch = t_epoch + current_progress
            temp_log = ""
            for i, imgs in enumerate(dataloader):
                temp_log = self._train_one_batch(imgs, epoch, i, dataloader.__len__())
                    
            self.save_models(epoch)
            self._write_to_file(temp_log)
        
        return self.net_G, self.net_D;
    
    def _train_one_epoch(self, epoch, dataloader):
        temp_log = ""
        for i, imgs in enumerate(dataloader):
            temp_log = self._train_one_batch(imgs, epoch, i, dataloader.__len__())
        self.save_models(epoch)
        self._write_to_file(temp_log)

    def _cal_generator_loss(self, input, output, ground_truth, valid_label):
        adv_loss = self.adv_loss(self.net_D(torch.cat((input, output), 1)), valid_label)
        g_loss = self.opt.adv_lambda * adv_loss
        g_loss = g_loss + self.opt.l1_lambda * self.l1_loss(output, ground_truth)
        
        return adv_loss, g_loss;
    
    def _cal_discriminator_loss(self, ground_truth_distingue, fake_distingue, valid_label, fake_label):            
        real_loss = self.adv_loss(ground_truth_distingue, valid_label)
        fake_loss = self.adv_loss(fake_distingue, fake_label)
        
        return real_loss, fake_loss, (real_loss + fake_loss) / 2;
    
    def _train_one_batch(self, imgs, epoch, batch, total_batch):
        self.net_G.train()
        self.net_D.train()
        self.net_interpolate.eval()
        Tensor = torch.FloatTensor if not (self.opt.isCudaUsed and torch.cuda.is_available()) else torch.cuda.FloatTensor  # @UndefinedVariable
        
        pres = imgs[:, 0]
        lats = imgs[:, 2]
        mids = imgs[:, 1]
        
        # Adversarial ground truths
        valid = Variable(Tensor(mids.shape[0], 1, 1, 1).fill_(0.95), requires_grad=False)
        fake = Variable(Tensor(mids.shape[0], 1, 1, 1).fill_(0.1), requires_grad=False)

        # Configure input
        if (self.opt.isCudaUsed):
            inputPres = pres.to('cuda')
            inputLats = lats.to('cuda')
            expectedOutput = mids.to('cuda')
        else:
            inputPres = pres
            inputLats = lats
            expectedOutput = mids
         
        # -----------------
        #  Train Generator
        # -----------------
         
        self.optimizer_G.zero_grad()  # set G's gradient to zero
        imgs_inter = self.net_interpolate(inputPres.data, inputLats.data)  # get interpolated frame from the interpolation network
        imgs_ref = self.net_G(imgs_inter)  # get refinement frame
         
        # Calculate gradient for G
        # Loss measures generator's ability to fool the discriminator and generate similar image to ground truth
        adv_loss, g_loss = self._cal_generator_loss(imgs_inter, imgs_ref, expectedOutput, valid)
        g_loss.backward()

        self.optimizer_G.step()  # update G's weights

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()  # set D's gradient to zero

        # Calculate gradient for D
        gt_distingue = self.net_D(torch.cat((imgs_inter.detach(), expectedOutput), 1))
        fake_distingue = self.net_D(torch.cat((imgs_inter.detach(), imgs_ref.detach()), 1))
        real_loss, fake_loss, d_loss = self._cal_discriminator_loss(gt_distingue, fake_distingue, valid, fake)
        d_loss.backward()

        self.optimizer_D.step()  # update D's weights
                
        # Show progress
        psnr1 = utils.cal_psnr_tensor(imgs_inter.data[0].cpu(), expectedOutput.data[0].cpu())
        psnr2 = utils.cal_psnr_tensor(imgs_ref.data[0].cpu(), expectedOutput.data[0].cpu())
        temp_log = ("V4: [Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f] [psnr1: %f] [psnr_ref: %f]" 
                    % (epoch, batch, total_batch, d_loss.item(), g_loss.item(), psnr1, psnr2))
        if (batch % 100 == 0):
            print(temp_log)
         
        # Display result (input and output) after every opt.sample_intervals
        batches_done = epoch * total_batch + batch
        if batches_done % self.opt.sample_interval == 0:
            save_image(imgs_ref.data[:25], self.opt.path + "/l_%d.png" % batches_done, nrow=5, normalize=True)
            print("Saved l_%d.png" % batches_done)
            print(temp_log)
            
        return temp_log
        
    def save_models(self, epoch, output_path=None):
        '''
        Save model into file which contains state's information
        :param epoch: last epochth train
        :param output_path: saved directory path
        '''
        outpath = self.opt.default_model_path if output_path is None else output_path
        os.makedirs(outpath, exist_ok=True)
        
        state_gen = {'epoch': epoch + 1,
                     'state_dict': self.net_G.state_dict(),
                     'optimizer': self.optimizer_G.state_dict(),
                     }
        state_dis = {'epoch': epoch + 1,
                     'state_dict': self.net_D.state_dict(),
                     'optimizer': self.optimizer_D.state_dict(),
                     }
        
        torch.save(state_gen, outpath + "/pix2pix_gen_" + self.opt.path + ".pt");
        torch.save(state_dis, outpath + "/pix2pix_dis_" + self.opt.path + ".pt");
        
        os.makedirs("%s/cpt" % (outpath), exist_ok=True)
        torch.save(state_gen, "%s/cpt/pix2pix_gen_%s_%d.pt" % (outpath, self.opt.path, epoch));
        return;
    
    def _load_model(self, model, path):
        '''
        Load network's state data (respecting to saved information)
        :param model:
        :param optimizer:
        :param path: file path
        '''
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if self.opt.isCudaUsed: model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if not self.opt.isCudaUsed:
            return start_epoch
        
        # copy tensor into GPU manually
        for state in optimizer.state.values():
            for k, v in state.items():                
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                    
        return start_epoch
    
    def _write_to_file(self, content=None):
        path = self.opt.path + "/readme.txt"
        
        if content is not None:
            utils.write_to_existed_text_file(path, "\n" + content)
            return;
        
        s = ""
        s += str(self.opt)
        s += "\n"
        s += str(self.net_G)
        s += "\n"
        s += str(self.net_D)
        s += "\n"
        s += ("Dataset: " + self.data_path)
        utils.write_to_text_file(path, s)


def initParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--crop_size", type=int, default=64, help="size of cropping area from image")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="number of batches between image sampling") 
    parser.add_argument("--test_interval", type=int, default=50, help="number of epochs between testing while training")
    parser.add_argument("--patch_size", type=int, default=3, help="the number of frames in a patch")
    parser.add_argument("--path", type=str, default="pix2pix_try", help="output folder")
    parser.add_argument("--default_model_path", type=str, default="models_test", help="the default path of saved models")
    parser.add_argument("--adv_lambda", type=float, default=1.0, help="the default weight of adv Loss")
    parser.add_argument("--l1_lambda", type=float, default=100.0, help="the default weight of L1 Loss")
    parser.add_argument("--isCudaUsed", type=bool, default=True, help="run with GPU or CPU (default)")
    parser.add_argument("--gen_load", type=str, default=None, help="loaded generator for training")
    parser.add_argument("--dis_load", type=str, default=None, help="loaded discriminator for training")
    
    return parser.parse_args()


def main():
    print("From train_pix2pix module...")
    opt = initParameters()
    nets = pix2pix(opt)
    nets.train(opt)


main()
