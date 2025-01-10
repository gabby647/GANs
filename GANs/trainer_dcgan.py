# %%
"""
wgan with different loss function, used the pure dcgan structure.
"""
import os
import time
import torch
import datetime

import torch.nn as nn
import torchvision

import numpy as np

import sys

sys.path.append('.')
sys.path.append('..')

from models.wgan_gp_beysian import Generator, Discriminator
from utils.utils import *

ngpu = 0

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netD = Discriminator(ngpu).to(device)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))




def calc_gradient_penalty(netD, real_data, fake_data):
    # print real_data.size()
    alpha = torch.rand(1, 1)  ##返回一个tensor
    alpha = alpha.expand(real_data.size())  ##扩展数据维度
    alpha = alpha.cuda(ngpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(ngpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)  ###计算微分

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 5
    return gradient_penalty


# %%
class Trainer_dcgan(object):
    def __init__(self, data_loader, config):
        super(Trainer_dcgan, self).__init__()

        # data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model

        # model hyper-parameters
        self.imsize = config.img_size
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.channels = config.channels
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lambda_gp = config.lambda_gp

        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard

        # path
        self.image_path = config.dataroot
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.version = config.version

        # step
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # path with version
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()

    def train(self):
        '''
        Training
        '''

        # fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))  # （*, 100）
        end = time.time()

        for epoch in range(self.epochs):
            # start time
            start_time = time.time()

            for i, (real_images, _) in enumerate(self.data_loader):

                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                # configure input
                real_images = tensor2var(real_images)

                self.D.train()
                self.G.train()
                # ==================== Train D ==================
                # train D more iterations than G

                self.D.zero_grad(True)

                # compute loss with real images
                d_out_real = self.D(real_images)

                d_loss_real = - torch.mean(d_out_real)

                # noise z for generator
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))  # 64, 100

                fake_images = self.G(z)  # (*, c, 64, 64)
                d_out_fake = self.D(fake_images)  # (*,)

                d_loss_fake = torch.mean(d_out_fake)

                # total d loss
                d_loss = d_loss_real + d_loss_fake

                # for the wgan loss function
                # gradient_penalty = calc_gradient_penalty(self.D, real_images, fake_images)
                # d_loss = gradient_penalty + d_loss

                d_loss.backward()

                # update D
                self.d_optimizer.step()

                # train the generator every 5 steps
                if i % self.g_num == 0:

                    # =================== Train G and gumbel =====================
                    for p in self.D.parameters():
                        p.requires_grad = False  # to avoid computation

                    self.G.zero_grad()
                    # create random noise
                    fake_images = self.G(z)

                    # compute loss with fake images
                    g_out_fake = self.D(fake_images)  # batch x n

                    g_loss_fake = - torch.mean(g_out_fake)

                    g_loss_fake.backward()
                    # update G
                    self.g_optimizer.step()



            # print out log info
            if (epoch) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, "
                      .format(elapsed, epoch, self.epochs, epoch,
                              self.epochs, d_loss.item(), g_loss_fake.item()))

            # sample images
            if (epoch) % self.sample_step == 0:
                self.G.eval()
                # save real image
                save_sample(self.sample_path + '/real_images/', real_images, epoch)

                with torch.no_grad():
                    fake_images = self.G(fixed_z)
                    # save fake image
                    save_sample(self.sample_path + '/fake_images/', fake_images, epoch)

                # sample sample one images
                # for the FID score
                # self.number = save_sample_one_image(self.sample_path, real_images, fake_images, epoch)

            # save model checkpoint
            if (epoch) % self.model_save_step == 0:
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': self.G.state_dict(),
                    'g_loss': g_loss_fake,
                    'D_state_dict': self.D.state_dict(),
                    'd_loss': d_loss,
                },
                    os.path.join(self.model_save_path, '{}.pth.tar'.format(epoch))
                )

    def build_model(self):

        self.G = Generator(image_size=self.imsize, z_dim=self.z_dim, conv_dim=self.g_conv_dim,
                           channels=self.channels).cuda()
        self.D = Discriminator(image_size=self.imsize, conv_dim=self.d_conv_dim, channels=self.channels).cuda()

        # apply the weights_init to randomly initialize all weights
        # to mean=0, stdev=0.2
        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        # optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)
