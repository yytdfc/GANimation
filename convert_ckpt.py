#!/usr/bin/env python
# encoding: utf-8

import torch
# import copy
# g = torch.load('checkpoints/net_epoch_52_id_G.pth')
# g['img_reg.6.weight'] = g.pop('img_reg.0.weight')
# g['attetion_reg.6.weight'] = g.pop('attetion_reg.0.weight')
# for k in g.keys():
    # if 'main.' in k:
        # ks = k.split('.')
        # if int(ks[1]) >= 15:
            # ks[1] = str(int(ks[1]) - 15)
            # ks[0] = 'img_reg'
            # k2 = '.'.join(ks)
            # print('%s -> %s' % (k, k2))
            # g[k2] = copy.deepcopy(g[k])
            # ks[0] = 'attetion_reg'
            # k2 = '.'.join(ks)
            # print('%s -> %s' % (k, k2))
            # g[k2] = copy.deepcopy(g[k])
            # del g[k]

# torch.save(g, 'checkpoints/st/net_epoch_52_id_G.pth')


# o = torch.load('checkpoints/st/opt_epoch_56_id_G.pth.old')
# print(o.keys())
# print(o['state'].keys())
# torch.save(g, 'checkpoints/st/opt_epoch_56_id_G.pth')

def remove_sn(layer):
    if type(layer) is gen.ResidualBlock:
        for l in layer.main:
            remove_sn(l)
        return
    try:
        nn.utils.remove_spectral_norm(layer)
    except ValueError:
        pass

def set_sn(layer):
    if type(layer) is gen.ResidualBlock:
        for l in layer.main:
            set_sn(l)
        return
    if type(layer) is nn.Conv2d or \
       type(layer) is nn.ConvTranspose2d:
        try:
            nn.utils.spectral_norm(layer)
        except KeyError:
            pass

import networks.generator_wasserstein_gan as gen
import torch.nn as nn
ge = gen.Generator(c_dim=17)
for layer in ge.main:
    remove_sn(layer)
# for layer in ge.img_reg:
    # remove_sn(layer)
# for layer in ge.attetion_reg:
    # remove_sn(layer)
g = torch.load('checkpoints/test/net_epoch_52_id_G.pth')
ge.load_state_dict(g)
print('loaded')
for layer in ge.main:
    set_sn(layer)
# for layer in ge.img_reg[:5]:
    # set_sn(layer)
# for layer in ge.attetion_reg[:5]:
    # set_sn(layer)
print(ge.state_dict().keys())
print(g.keys())
ge2 = gen.Generator(c_dim=17)
print(ge2.state_dict().keys())
ge2.load_state_dict(ge.state_dict())
torch.save(ge2.state_dict(), 'checkpoints/st/net_epoch_1_id_G.pth')


import networks.discriminator_wasserstein_gan as dis
import torch.nn as nn
ge = dis.Discriminator(c_dim=17)
for layer in ge.main:
    remove_sn(layer)
g = torch.load('checkpoints/test/net_epoch_52_id_D.pth')
ge.load_state_dict(g)
print('loaded')
for layer in ge.main:
    set_sn(layer)
print(ge.state_dict().keys())
torch.save(ge.state_dict(), 'checkpoints/st/net_epoch_1_id_D.pth')
