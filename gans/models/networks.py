# -*- coding: utf-8 -*-
'''
# Created on 2021/01/18 17:09:16
# @filename: networks.py
# @author: tcxia
'''

from utils.util import get_norm_layer, init_net
from utils.resnet import ResnetGenerator
from utils.unet import UnetGenrator
from utils.nlayer import NLayerDiscriminator
from utils.pix2pix import PixDiscriminator


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9':
        net = ResnetGenerator(input_nc,
                              output_nc,
                              ngf,
                              norm_layer=norm_layer,
                              use_dropout=use_dropout,
                              n_blocks=9)
    elif netG == 'resnet_6':
        net = ResnetGenerator(input_nc,
                              output_nc,
                              ngf,
                              norm_layer=norm_layer,
                              use_dropout=use_dropout,
                              n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenrator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenrator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recongized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_layer=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':
        net = PixDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    
    return init_net(net, init_type, init_gain, gpu_ids)
