#!/usr/bin/env python
# encoding: utf-8

import torch
import copy
g = torch.load('checkpoints/net_epoch_52_id_G.pth')
g['img_reg.6.weight'] = g.pop('img_reg.0.weight')
g['attetion_reg.6.weight'] = g.pop('attetion_reg.0.weight')
for k in g.keys():
    if 'main.' in k:
        ks = k.split('.')
        if int(ks[1]) >= 15:
            ks[1] = str(int(ks[1]) - 15)
            ks[0] = 'img_reg'
            k2 = '.'.join(ks)
            print('%s -> %s' % (k, k2))
            g[k2] = copy.deepcopy(g[k])
            ks[0] = 'attetion_reg'
            k2 = '.'.join(ks)
            print('%s -> %s' % (k, k2))
            g[k2] = copy.deepcopy(g[k])
            del g[k]

torch.save(g, 'checkpoints/st/net_epoch_52_id_G.pth')


# o = torch.load('checkpoints/st/opt_epoch_56_id_G.pth.old')
# print(o.keys())
# print(o['state'].keys())
# torch.save(g, 'checkpoints/st/opt_epoch_56_id_G.pth')


# import networks.generator_wasserstein_gan as gen
# ge = gen.Generator()
# ge.load_state_dict(g)

