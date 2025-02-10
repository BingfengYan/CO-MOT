# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import copy
import numpy as np
import collections

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(f'loaded {model_path}')
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you set the correct --num_classes for your own dataset.'
    state_dict_old = copy.deepcopy(state_dict)
    for k in state_dict_old:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                if 'class_embed' in k:
                    print("load class_embed: {} shape={}".format(k, state_dict[k].shape))
                    if model_state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        state_dict[k] = state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        state_dict[k] = state_dict[k][1:4]
                    elif model_state_dict[k].shape[0] == 11:
                        state_dict[k] = state_dict[k][1:12]
                    elif model_state_dict[k].shape[0] == 100:
                        state_dict[k] = state_dict[k].repeat_interleave(model_state_dict[k].shape[0]//state_dict[k].shape[0]+1, dim=0)[:model_state_dict[k].shape[0]]
                    elif model_state_dict[k].shape[0] == 91 and state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k].repeat_interleave(91, dim=0)
                    elif model_state_dict[k].shape[0] == 2000:
                        state_dict[k] = state_dict[k].repeat_interleave(model_state_dict[k].shape[0]//state_dict[k].shape[0]+1, dim=0)[:model_state_dict[k].shape[0]]
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                    continue
                state_dict[k] = model_state_dict[k]
        elif k.replace('in_proj_weight', 'in_proj.weight') in model_state_dict:
            k_dst = k.replace('in_proj_weight', 'in_proj.weight')
            print('{}->{}'.format(k, k_dst))
            state_dict = collections.OrderedDict([(k_dst, v) if k_ == k else (k_, v) for k_, v in state_dict.items()])
        elif k.replace('in_proj_bias', 'in_proj.bias') in model_state_dict:
            k_dst = k.replace('in_proj_bias', 'in_proj.bias')
            print('{}->{}'.format(k, k_dst))
            state_dict = collections.OrderedDict([(k_dst, v) if k_ == k else (k_, v) for k_, v in state_dict.items()])
        elif 'transformer.decoder.layers' in k and 'self_attn.in_proj' in k:
            k_dst_q = k.replace('in_proj_', 'in_proj_q.')
            k_dst_k = k.replace('in_proj_', 'in_proj_k.')
            k_dst_v = k.replace('in_proj_', 'in_proj_v.')
            print('{}->({},{},{})'.format(k, k_dst_q, k_dst_k, k_dst_v))
            state_dict[k_dst_q], state_dict[k_dst_k], state_dict[k_dst_v] = torch.chunk(state_dict[k], 3, dim=0)
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):  # pretrain model
            if 'decoder_two' in k:
                state_dict[k] = state_dict[k.replace('.decoder_two.', '.decoder.')]
            elif '_embed_two' in k:
                state_dict[k] = state_dict[k.replace('_embed_two.', '_embed.')]
            else:
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model



