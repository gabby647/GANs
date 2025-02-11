# %%
import os

from trainer_tuxiang import Trainer_dcgan
from utils.utils import *
from dataset.dataset import getdDataset

# set the gpu number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse


# %%
def get_parameters():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='wgan_gp_beysian_2', choices=['wgan-gp-resblock-attention,wgan-gp, LSGAN, dcgan, wgan-gp-attention, cgan, lsgan, SAGAN, VQGAN'])
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--g_num', type=int, default=5, help='train the generator every 5 steps')
    parser.add_argument('--z_dim', type=int, default=100, help='noise dim')
    parser.add_argument('--c_dim', type=int, default=100, help='latent dim')
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--version', type=str, default='test', help='the version of the path, for implement')

    # Training setting
    parser.add_argument('--epochs', type=int, default=1001, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--g_lr', type=float, default=0.0004, help='use TTUR lr rate for Adam')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='use TTUR lr rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_gp', type=float, default=10, help='lambda for wgan gp')

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='tr', choices=['mnist', 'cifar10', 'fashion'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False, help='use tensorboard to record the loss')

    # Path
    parser.add_argument('--dataroot', type=str, default='../data', help='dataset path')
    parser.add_argument('--log_path', type=str, default='./logs', help='the output log path')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint', help='model save path')
    parser.add_argument('--sample_path', type=str, default='./samples', help='the generated sample saved path')

    # Step size
    parser.add_argument('--log_step', type=int, default=20, help='every default{10} epoch save to the log')
    parser.add_argument('--sample_step', type=int, default=20,
                        help='every default{100} epoch save the generated images and real images')
    parser.add_argument('--model_save_step', type=int, default=20)

    return parser.parse_args()


# %%

def main(config):
    # data loader
    data_loader = getdDataset(config)

    # delete the exists path
    del_folder(config.sample_path, config.version)
    del_folder(config.log_path, config.version)
    del_folder(config.model_save_path, config.version)

    # create directories if not exist
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.model_save_path, config.version)

    # save sample images
    make_folder(config.sample_path, config.version + '/real_images')
    make_folder(config.sample_path, config.version + '/fake_images')

    if config.train:
        if config.model == 'wgan_gp_beysian_2':
            trainer = Trainer_dcgan(data_loader, config)
        trainer.train()


# %%

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
