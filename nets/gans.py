'''
Created on Apr 10, 2020

@author: Quang TRAN
'''

import os
import sys

import torch.cuda
import torch.optim
import torch.nn as nn
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader

import nets.generators as generators
import nets.discriminators as discriminators
import utils.utils as utils
import data.dataset_handler as dataset_handler
import utils.loss as loss


class GenNet():

    def __init__(self, opt):
        self.opt = opt
        self.path = "./train/" + self.opt.path
        self.data_path = dataset_handler.Dataset.D_UCF101_BODY_TRAIN.value
        self.test_data_path = dataset_handler.Dataset.D_UCF101_BODY_TEST.value
        self.generator = generators.GeneratorWithCondition_NoNoise_V7(self.opt)
        self.discriminator = discriminators.DisCriminatorWithCondition_V2(self.opt)
        self.cudaUsed = torch.cuda.is_available() and self.opt.isCudaUsed
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.MSELoss()
        self.gd_loss = loss.GDL(cudaUsed=self.cudaUsed)
        self.ms_ssim = loss.MS_SSIM()
        self.data_loader = None
        self.testing_dataset = None
        
        if self.cudaUsed:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.adversarial_loss = nn.BCEWithLogitsLoss().cuda()
            self.l1_loss = nn.MSELoss().cuda()
            self.gd_loss = loss.GDL(cudaUsed=self.cudaUsed).cuda()
            self.ms_ssim = loss.MS_SSIM().cuda()
        
        # Initialize weights
        self.generator.apply(self._weights_init_normal)
        self.discriminator.apply(self._weights_init_normal)
        
        # Optimizers
        self._optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        self._optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

    def _weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    def _write_to_file(self, content=None):
        path = self.path + "/readme.txt"
        
        if content is not None:
            utils.write_to_existed_text_file(path, "\n" + content)
            return;
        
        s = ""
        s += str(self.opt)
        s += "\n"
        s += str(self.generator)
        s += "\n"
        s += str(self.discriminator)
        s += "\n"
        s += ("Dataset: " + self.data_path)
        utils.write_to_text_file(path, s)
        
    def trainFromTrainedModel(self, gen_net_path, dis_net_path, n_epochs=0):
        print('Loading pre-trained model ...')
        epoch = self.load_gen_model(gen_net_path, isTraining=True)
        self.load_dis_model(dis_net_path, isTraining=True)
        epoch = n_epochs if epoch == -1 else epoch
        
        return self.trainGAN(current_progress=epoch)
    
    def load_training_dataset(self):
        if self.data_loader is not None:
            return self.data_loader
        
        print("Loading training dataset: %s" % self.data_path)
        dataset = dataset_handler.ImageDatasetLoader(self.opt).loadImageDatasetForNoNoise(self.data_path);
        self.data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        print("Done.")
        
        return self.data_loader    
        
    def load_testing_dataset(self):
        if self.testing_dataset is not None:
            return self.testing_dataset
        
        print("Loading testing dataset: %s" % self.test_data_path)
        self.testing_dataset = dataset_handler.ImageDatasetLoader(self.opt).loadFirstPatchesWithFullImg(self.test_data_path, no_patches=-1)
        print("Done.")
        
        return self.testing_dataset

    def trainGAN(self, current_progress=0):
        '''
        Train GAN model
        :param current_progress: use in case that train on trained model
        '''
        print(self.opt);
        print(self.generator)
        print(self.discriminator)
        
        self.generator.train()
        self.discriminator.train()

        self.load_training_dataset()
        self.load_testing_dataset()
        
        if current_progress <= 0:
            no_epochs = self.opt.n_epochs
            current_progress = 0
        else:
            no_epochs = self.opt.n_epochs - current_progress
        
        self._write_to_file()
        
        # ----------
        #  Training
        # ----------    
        for t_epoch in range(no_epochs):
            epoch = t_epoch + current_progress
            temp_log = ""
            for i, imgs in enumerate(self.data_loader):
                temp_log = self._train_one_batch(imgs, epoch, i, dataloader.__len__())
                    
            self.save_models(epoch)
            self._write_to_file(temp_log)
            
            if (self.opt.test_interval > 0 and epoch % self.opt.test_interval == 0) : training_observer.notify(self, epoch, multi_thread=True)
        
        return self.generator, self.discriminator;
    
    def _train_one_epoch(self, epoch, dataloader):
        temp_log = ""
        for i, imgs in enumerate(dataloader):
            temp_log = self._train_one_batch(imgs, epoch, i, dataloader.__len__())
                    
        self.save_models(epoch)
        self._write_to_file(temp_log)
    
    def _train_one_batch(self, imgs, epoch, batch, total_batch):
        Tensor = torch.FloatTensor if not (self.opt.isCudaUsed and torch.cuda.is_available()) else torch.cuda.FloatTensor  # @UndefinedVariable
        
        pres = imgs[:, 0]
        lats = imgs[:, 2]
        mids = imgs[:, 1]
        
        # Adversarial ground truths
        valid = Variable(Tensor(mids.shape[0], 1, 1, 1).fill_(0.95), requires_grad=False)
        fake = Variable(Tensor(mids.shape[0], 1, 1, 1).fill_(0.1), requires_grad=False)
                
        # Configure input
        if (self.cudaUsed):
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
         
        self._optimizer_G.zero_grad()  # set G's gradient to zero
        gen_imgs = self.generator(inputPres.data, inputLats.data)  # generate output images
         
        # Calculate gradient for G
        # Loss measures generator's ability to fool the discriminator and generate similar image to ground truth
        adv_loss, g_loss = self._cal_generator_loss(gen_imgs, expectedOutput, valid)
        g_loss.backward()
                
        self._optimizer_G.step()  # update G's weights
         
        # ---------------------
        #  Train Discriminator
        # ---------------------
         
        self._optimizer_D.zero_grad()  # set D's gradient to zero
                
        # Calculate gradient for D
        gt_distingue = self.discriminator(expectedOutput)
        fake_distingue = self.discriminator(gen_imgs.detach())
        real_loss, fake_loss, d_loss = self._cal_discriminator_loss(gt_distingue, fake_distingue, valid, fake)
        d_loss.backward()
                
        self._optimizer_D.step()  # update D's weights
                
        # Show progress
        psnr = utils.cal_psnr_tensor(gen_imgs.data[0].cpu(), expectedOutput.data[0].cpu())
        temp_log = ("V3: [Epoch %d] [Batch %d/%d] [D loss (real/fake): (%f, %f)] [G loss (adv_loss): %f (%f)] [psnr: %f]" 
                    % (epoch, batch, total_batch, real_loss.item(), fake_loss.item(), g_loss.item(), adv_loss.item(), psnr))
        if (batch % 100 == 0):
            print(temp_log)
         
        # Display result (input and output) after every opt.sample_intervals
        batches_done = epoch * total_batch + batch
        if batches_done % self.opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], self.path + "/l_%d.png" % batches_done, nrow=5, normalize=True)
            print("Saved l_%d.png" % batches_done)
            print(temp_log)
            
        return temp_log
    
    def _cal_generator_loss(self, output, ground_truth, valid_label):
        adv_loss = self.adversarial_loss(self.discriminator(output), valid_label)
        g_loss = self.opt.adv_lambda * adv_loss
        g_loss = g_loss + self.opt.l1_lambda * self.l1_loss(output, ground_truth)
        g_loss = g_loss + self.opt.gdl_lambda * self.gd_loss(output, ground_truth)
        g_loss = g_loss + self.opt.ms_ssim_lambda * self.ms_ssim(output, ground_truth)
            
        return adv_loss, g_loss;
    
    def _cal_discriminator_loss(self, ground_truth_distingue, fake_distingue, valid_label, fake_label):            
        real_loss = self.adversarial_loss(ground_truth_distingue, valid_label)
        fake_loss = self.adversarial_loss(fake_distingue, fake_label)
        
        return real_loss, fake_loss, (real_loss + fake_loss) / 2;
        
    def save_models(self, epoch, output_path=None):
        '''
        Save model into file which contains state's information
        :param epoch: last epochth train
        :param output_path: saved directory path
        '''
        outpath = self.opt.default_model_path if output_path is None else output_path
        os.makedirs(outpath, exist_ok=True)
        
        state_gen = {'epoch': epoch + 1,
                     'state_dict': self.generator.state_dict(),
                     'optimizer': self._optimizer_G.state_dict(),
                     }
        state_dis = {'epoch': epoch + 1,
                     'state_dict': self.discriminator.state_dict(),
                     'optimizer': self._optimizer_D.state_dict(),
                     }
        
        torch.save(state_gen, outpath + "/gennet_gen_" + self.opt.path + ".pt");
        torch.save(state_dis, outpath + "/gennet_dis_" + self.opt.path + ".pt");
        
        os.makedirs("%s/cpt" % (outpath), exist_ok=True)
        torch.save(state_gen, "%s/cpt/gennet_gen_%s_%d.pt" % (outpath, self.opt.path, epoch));
        return;
    
    def load_gen_model(self, path="/gennet_gen.pt", isTraining=False):
        target = (self.opt.default_model_path + path) if not isTraining else path
        try:
            return self._load_model(self.generator, target)
        except TypeError:
            self.generator = torch.load(target)
            return -1;
        
    def load_dis_model(self, path="/gennet_dis.pt", isTraining=False):
        target = self.opt.default_model_path + path if not isTraining else path
        try:
            return self._load_model(self.discriminator, target)
        except TypeError:
            # For old version
            self.discriminator = torch.load(target)
            print(sys.exc_info()[0])
            return -1;
                    
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
        
        if self.cudaUsed: model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if not self.cudaUsed:
            return start_epoch
        
        # copy tensor into GPU manually
        for state in optimizer.state.values():
            for k, v in state.items():                
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                    
        return start_epoch
    
    def load_gen_for_evaluation(self, path):
        '''
        Load model for evaluation, support old loading approach
        :param path: "/gen_30.pt"
        '''
        n_epoch = self.load_gen_model(path, isTraining=False)
            
        self.eval()
        return n_epoch, self.generator
        
    def generate_frame(self, input1, intput2):
        return self.generator.forward(input1, intput2)
    
    def eval(self):
        '''
        Set generator into evaluation mode
        '''
        self.generator.eval()
