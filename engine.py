# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.distributed as dist
import torch
import util.misc as utils

from datasets.data_prefetcher import data_dict_to_cuda
attr_dict = dict()
attr_dict["categories"] = {
    0: {"supercategory": "none", "id": 0, "name": "pedestrian"},
    1: {"supercategory": "none", "id": 1, "name": "bicycle"},
    2: {"supercategory": "none", "id": 2, "name": "car"},
    3: {"supercategory": "none", "id": 3, "name": "motorcycle"},
    5: {"supercategory": "none", "id": 5, "name": "bus"},
    6: {"supercategory": "none", "id": 6, "name": "train"},
    7: {"supercategory": "none", "id": 7, "name": "truck"},
    90: {"supercategory": "none", "id": 90, "name": "rider"},
    91: {"supercategory": "none", "id": 91, "name": "other person"},
    92: {"supercategory": "none", "id": 92, "name": "trailer"},
    93: {"supercategory": "none", "id": 93, "name": "other vehicle"}
}

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    iter_num = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if True: 
            loss_two = 0
            loss_ori = 0
            for k, v in loss_dict_reduced_scaled.items():
                if '_two_' in k: loss_two += v
                else: loss_ori += v
            loss_dict_reduced_scaled['loss_ori'] = loss_ori
            loss_dict_reduced_scaled['loss_two'] = loss_two
            # if loss_two > 0:
            #     losses /= 2.0  # 由于多加了一倍的loss，因此这里减掉
            #     loss_value /= 2.0
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        if torch.isnan(grad_total_norm).any():
            print(data_dict['gt_instances'])
            optimizer.zero_grad()
            
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'same_num_dict'):
                    # if True:
            same_num_dict = utils.reduce_dict(criterion.same_num_dict, average=False)
            same = 0
            all = 0
            for k, v in same_num_dict.items():
                if '_same' in k: same += v
                else: all += v
            if all > 0:
                same_num_dict['ratio'] = same * 1.0 / all
            metric_logger.update(loss=loss_value, **dict(loss_dict_reduced_scaled.items(), **same_num_dict))
        else:
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

