# %%
'''
pure dcgan structure.
code similar sample from the pytorch code.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn

import numpy as np


class ConditionalEmbedding(nn.Module):
    def __init__(self, c_dim, embedding_dim):
        super(ConditionalEmbedding,self).__init__()
        self.embedding = nn.Linear(c_dim, embedding_dim)

    def forward(self, c):
        reward = self.embedding(c)
        re = torch.relu_(reward)
        return re



# %%
class Generator(nn.Module):
    '''
    pure Generator structure

    '''

    def __init__(self, image_size=128, z_dim=100, c_dim=100, conv_dim=64, channels=3):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.embedding = ConditionalEmbedding(4096, self.z_dim)

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.l1 = nn.Sequential(
            # input is Z, going into a convolution.
            nn.ConvTranspose2d(self.z_dim + self.c_dim, conv_dim * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        z = z.view(z.size(0), -1, 1, 1)  # (*, 100, 1, 1)
        # c_embedded = self.embedding(c)
        c_embedded = self.embedding(c).view(c.size(0), -1, 1, 1)  # 将c_embedded调整为[1, 100, 1, 1]

        x = torch.cat([z, c_embedded], dim=1)
        out = self.l1(x)

        out = self.l2(out)

        out = self.l3(out)

        out = self.l4(out)

        out = self.l5(out)

        out = self.last(out)

        return out


# %%
class Discriminator(nn.Module):
    '''
    pure discriminator structure

    # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
    # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
    # Image (Cx64x64)
    '''

    def __init__(self, image_size=128,  conv_dim=64, channels=3):
        super(Discriminator, self).__init__()
        self.channels = channels


        curr_ims = image_size // 2

        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            nn.LayerNorm([conv_dim, curr_ims, curr_ims]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim
        curr_ims = curr_ims // 2

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([curr_dim * 2, curr_ims, curr_ims]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        curr_ims = curr_ims // 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LayerNorm([curr_dim * 2, curr_ims, curr_ims]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        curr_ims = curr_ims // 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LayerNorm([curr_dim * 2, curr_ims, curr_ims]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        curr_ims = curr_ims // 2

        self.l5 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LayerNorm([curr_dim * 2, curr_ims, curr_ims]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2

        # output layers
        self.last_adv = nn.Sequential(
            # The output of D is no longer a probability,
            # we do not apply sigmoid at the output of D.
            nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        out = self.l1(x)  # (*, 64, 32, 32)

        out = self.l2(out)  # (*, 128, 16, 16)

        out = self.l3(out)  # (*, 256, 8, 8)

        out = self.l4(out)  # (*, 512, 4, 4)

        out = self.l5(out)

        validity = self.last_adv(out)  # (*, 1, 1, 1)

        return validity.squeeze()


# %%
import sys

sys.path.append('..')
import pprint
from torchinfo import summary

if __name__ == "__main__":
    G = Generator().cuda()
    D = Discriminator().cuda()

    # get model summary as string
    G_model_stats = summary(G, (64, 100))
    D_model_stats = summary(D, (64, 1, 64, 64))

    pprint.pprint(G_model_stats)
    pprint.pprint(D_model_stats)

    summary_str = (G_model_stats), (D_model_stats)
    with open('model_structure.log', 'w') as tf:
        pprint.pprint(summary_str, stream=tf)
