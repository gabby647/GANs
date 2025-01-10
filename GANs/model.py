from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchlib

from torch.autograd import grad


# ==============================================================================
# =                                loss function                               =
# ==============================================================================

def get_losses_fn(mode):
    if mode == 'gan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.nn.functional.binary_cross_entropy_with_logits(r_logit, torch.ones_like(r_logit))
            f_loss = torch.nn.functional.binary_cross_entropy_with_logits(f_logit, torch.zeros_like(f_logit))
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = torch.nn.functional.binary_cross_entropy_with_logits(f_logit, torch.ones_like(f_logit))
            return f_loss

    elif mode == 'lsgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.nn.functional.mse_loss(r_logit, torch.ones_like(r_logit))
            f_loss = torch.nn.functional.mse_loss(f_logit, torch.zeros_like(f_logit))
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = torch.nn.functional.mse_loss(f_logit, torch.ones_like(f_logit))
            return f_loss

    elif mode == 'wgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = -r_logit.mean()
            f_loss = f_logit.mean()
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = -f_logit.mean()
            return f_loss

    elif mode == 'hinge_v1':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
            f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = torch.max(1 - f_logit, torch.zeros_like(f_logit)).mean()
            return f_loss

    elif mode == 'hinge_v2':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
            f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = - f_logit.mean()
            return f_loss

    else:
        raise NotImplementedError

    return d_loss_fn, g_loss_fn


# ==============================================================================
# =                                   others                                   =
# ==============================================================================

def gradient_penalty(f, real, fake, mode):
    device = real.device

    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = torch.rand(a.size()).to(device)
                b = a + 0.5 * a.std() * beta
            shape = [a.size(0)] + [1] * (a.dim() - 1)
            alpha = torch.rand(shape).to(device)
            inter = a + alpha * (b - a)
            return inter

        x = torch.tensor(_interpolate(real, fake), requires_grad=True)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        g = grad(pred, x, grad_outputs=torch.ones(pred.size()).to(device), create_graph=True)[0].view(x.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

        return gp

    if mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'none':
        gp = torch.tensor(0.0).to(device)
    else:
        raise NotImplementedError

    return gp


# ==============================================================================
# =                                    utils                                   =
# ==============================================================================

def _get_norm_fn_2d(norm):  # 2d
    if norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return nn.InstanceNorm2d
    elif norm == 'none':
        return torchlib.NoOp
    else:
        raise NotImplementedError


def _get_weight_norm_fn(weight_norm):
    if weight_norm == 'spectral_norm':
        return torch.nn.utils.spectral_norm
    elif weight_norm == 'weight_norm':
        return torch.nn.utils.weight_norm
    elif weight_norm == 'none':
        return torchlib.identity
    else:
        return NotImplementedError

# ==============================================================================
# =                                 Attention BLOCKs                           =
# ==============================================================================

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
        # return out, attention

# ==============================================================================
# =                                 models CGAN                                =
# ==============================================================================

class GeneratorCGAN(nn.Module):

    def __init__(self, z_dim, c_dim, dim=128):
        super(GeneratorCGAN, self).__init__()

        def dconv_bn_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.ls = nn.Sequential(
            dconv_bn_relu(z_dim + c_dim, dim * 4, 4, 1, 0, 0),  # (N, dim * 4, 4, 4)
            dconv_bn_relu(dim * 4, dim * 2),  # (N, dim * 2, 8, 8)
            dconv_bn_relu(dim * 2, dim),   # (N, dim, 16, 16)
            dconv_bn_relu(dim, dim),    # (N, dim, 32, 32)
            dconv_bn_relu(dim, dim),    # (N, dim, 64, 64)
            nn.ConvTranspose2d(dim, 3, 4, 2, padding=1), nn.Tanh()  # (N, 3, 128, 128)
        )

    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        x = torch.cat([z, c], 1)          ###将两个张量拼接在一起
        x = self.ls(x.view(x.size(0), x.size(1), 1, 1))
        return x
class GeneratorACGAN_znoise(nn.Module):

    def __init__(self, z_dim, c_dim, dim=128):
        super(GeneratorACGAN_znoise, self).__init__()

        def dconv_bn_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        norm_fn = _get_norm_fn_2d('none')
        weight_norm_fn = _get_weight_norm_fn('spectral_norm')

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.zcode = nn.Linear(c_dim, z_dim)
        self.ls = nn.Sequential(
            dconv_bn_relu(z_dim, dim, 4, 1, 0, 0))# (N, dim * 4, 4, 4)
        self.ls2 = nn.Sequential(
            dconv_bn_relu(dim * 2, dim * 2),  # (N, dim * 2, 8, 8)
            dconv_bn_relu(dim * 2, dim))   # (N, dim, 16, 16)

        self.ls3 = nn.Sequential(dconv_bn_relu(dim, dim))    # (N, dim, 32, 32)
        self.ls4 = nn.Sequential(dconv_bn_relu(dim, dim))    # (N, dim, 64, 64)
        self.ls5 = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, 4, 2, padding=1), nn.Tanh()  # (N, 3, 128, 128)
        )
        self.attn2 = Self_Attn(dim, 'relu')
        self.attn3 = Self_Attn(dim, 'relu')
        self.attn4 = Self_Attn(dim, 'relu')
    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        z_deconv = self.ls(z.view(z.size(0), z.size(1), 1, 1))
        condition_embedding = self.zcode(c)
        condition_deconv = self.ls(condition_embedding.view(condition_embedding.size(0), condition_embedding.size(1), 1, 1))
        # print("zsize, condition size", z_deconv.size(), condition_deconv.size())
        x = torch.cat([z_deconv, condition_deconv], 1)
        # print("xsize", x.size())
        x = self.ls2(x)
        x = self.attn2(x)
        x = self.ls3(x)
        x = self.attn3(x)
        x = self.ls4(x)
        # x = self.attn4(x)
        x = self.ls5(x)
        # print("outputsize", x.size())
        return x

class DiscriminatorCGAN(nn.Module):

    def __init__(self, x_dim, c_dim, dim=96, norm='none', weight_norm='spectral_norm'):
        super(DiscriminatorCGAN, self).__init__()

        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(  # (N, x_dim+c_dim, 32, 32) #128*128
            conv_norm_lrelu(x_dim + c_dim, 2*dim),
            # conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(2*dim, dim, stride=2),  # (N, dim , 16, 16)#64*64

            # conv_norm_lrelu(dim, dim * 2),
            # conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(dim, int(dim//2), stride=2),  # (N, dim*2, 8, 8) #32*32

            # conv_norm_lrelu(dim, dim),
            # conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(int(dim//2), int(dim//4), stride=2),  # (N, dim*2, 8, 8) #16*16

            # conv_norm_lrelu(dim * 2, dim * 2),
            # conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(int(dim//4), int(dim//6), stride=2),  # (N, dim*2, 8, 8)
            conv_norm_lrelu(int(dim//6), int(dim//6), stride=2),  # (N, 16, 4, 4)

            # conv_norm_lrelu(dim * 2, dim * 2),
            # conv_norm_lrelu(dim * 2, dim * 2),
            # conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8) #8*8

            # conv_norm_lrelu(dim * 2, dim * 2, kernel_size=3, stride=1, padding=0),
            # conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),
            # conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)
        )
        self.ls2 = nn.Sequential(conv_norm_lrelu(int(dim//6), 1, kernel_size=4, padding=0))  # (N, 16, 4, 4))
        # self.ls2 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
        #     torchlib.Reshape(-1, dim * 2),  # (N, dim*2)
        #     weight_norm_fn(nn.Linear(dim * 2, 1)),  # (N, 1),
        #     nn.Tanh()
        # )

    def forward(self, x, c):
        # x: (N, x_dim, 32, 32), c: (N, c_dim)
        c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), x.size(2), x.size(3)], dtype=c.dtype, device=c.device)
        logit = self.ls(torch.cat([x, c], 1))
        # print("discrimin:1:", logit.size())
        logit = self.ls2(logit)
        # print("discrimin:2:", logit.size())
        # logit = self.ls2(feature)
        return logit


# ==============================================================================
# =                           models Projection CGAN                           =
# ==============================================================================

GeneratorPCGAN = GeneratorCGAN


class DiscriminatorPCGAN(nn.Module):

    def __init__(self, x_dim, c_dim, dim=96, norm='none', weight_norm='spectral_norm'):
        super(DiscriminatorPCGAN, self).__init__()

        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(  # (N, x_dim, 32, 32)
            conv_norm_lrelu(x_dim, dim),
            conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim , 16, 16)

            conv_norm_lrelu(dim, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8)

            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=3, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)

            nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
            torchlib.Reshape(-1, dim * 2),  # (N, dim*2)
        )

        self.l_logit = weight_norm_fn(nn.Linear(dim * 2, 1))  # (N, 1)
        self.l_projection = weight_norm_fn(nn.Linear(dim * 2, c_dim))  # (N, c_dim)

    def forward(self, x, c):
        # x: (N, x_dim, 32, 32), c: (N, c_dim)
        feat = self.ls(x)
        logit = self.l_logit(feat)
        embed = (self.l_projection(feat) * c).mean(1, keepdim=True)
        logit += embed
        return logit


# ==============================================================================
# =                                models ACGAN                                =
# ==============================================================================

GeneratorACGAN = GeneratorCGAN


class DiscriminatorACGAN(nn.Module):

    def __init__(self, x_dim, c_dim, dim=96, norm='none', weight_norm='spectral_norm'):
        super(DiscriminatorACGAN, self).__init__()

        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(  # (N, x_dim, 32, 32) 128*128
            conv_norm_lrelu(x_dim, dim),
            # conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim , 16, 16)64*64

            # conv_norm_lrelu(dim, dim * 2),
            # conv_norm_lrelu(dim * 2, dim * 2),
            # conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8)32*32
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim*2, 8, 8)32*32
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim*2, 8, 8)16*16
            conv_norm_lrelu(int(dim), int(dim // 2), stride=2),  # (N, dim*2, 8, 8)8*8

            conv_norm_lrelu(int(dim // 2), int(dim // 2), kernel_size=3, stride=1, padding=0),
            conv_norm_lrelu(int(dim // 2), int(dim // 2), kernel_size=1, stride=1, padding=0),
            conv_norm_lrelu(int(dim // 2), int(dim // 2), kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)

            nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
            torchlib.Reshape(-1, int(dim // 2)),  # (N, dim*2)
        )

        self.l_gan_logit = weight_norm_fn(nn.Linear(int(dim // 2), 1))  # (N, 1)
        self.l_c_logit = nn.Linear(int(dim // 2), c_dim)  # (N, c_dim)

    def forward(self, x):
        # x: (N, x_dim, 32, 32)
        feat = self.ls(x)
        # print(feat.size())
        gan_logit = self.l_gan_logit(feat)
        l_c_logit = self.l_c_logit(feat)
        return gan_logit, l_c_logit


# ==============================================================================
# =                               models InfoGAN1                              =
# ==============================================================================

GeneratorInfoGAN1 = GeneratorACGAN
DiscriminatorInfoGAN1 = DiscriminatorACGAN


# ==============================================================================
# =                               models InfoGAN2                              =
# ==============================================================================

GeneratorInfoGAN2 = GeneratorACGAN


class DiscriminatorInfoGAN2(nn.Module):

    def __init__(self, x_dim, dim=96, norm='none', weight_norm='spectral_norm'):
        super(DiscriminatorInfoGAN2, self).__init__()

        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(  # (N, x_dim, 32, 32)
            conv_norm_lrelu(x_dim, dim),
            conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim , 16, 16)

            conv_norm_lrelu(dim, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8)

            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=3, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)

            nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
            torchlib.Reshape(-1, dim * 2),  # (N, dim*2)
            weight_norm_fn(nn.Linear(dim * 2, 1))  # (N, 1)
        )

    def forward(self, x):
        # x: (N, x_dim, 32, 32)
        logit = self.ls(x)
        return logit


class QInfoGAN2(nn.Module):

    def __init__(self, x_dim, c_dim, dim=96, norm='batch_norm', weight_norm='none'):
        super(QInfoGAN2, self).__init__()

        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                norm_fn(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(  # (N, x_dim, 32, 32)
            conv_norm_lrelu(x_dim, dim),
            conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(dim, dim, stride=2),  # (N, dim , 16, 16)

            conv_norm_lrelu(dim, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8)

            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=3, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),
            conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)

            nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
            torchlib.Reshape(-1, dim * 2),  # (N, dim*2)
            nn.Linear(dim * 2, c_dim)  # (N, c_dim)
        )

    def forward(self, x):
        # x: (N, x_dim, 32, 32)
        logit = self.ls(x)
        return logit
