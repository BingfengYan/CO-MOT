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
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .memory_bank import build_memory_bank
from .deformable_transformer_plus import build_deforamble_transformer, pos2posemb
from .qim import build as build_query_interaction_layer
from .deformable_detr import SetCriterion, MLP, sigmoid_focal_loss


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def initialize_for_single_clip(self, gt_instances: List[Instances]):  # 训练过程中每个视频段之前调用，传入GT值
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        gt_instances_i = self.gt_instances[self._current_frame_idx]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances_i],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(self._current_frame_idx, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J) * self.num_classes
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        i, j = torch.where(track_instances.obj_idxes[:, None] == obj_idxes)  # 获取跟踪query与之相同ID的对应索引
        track_instances.matched_gt_idxes[i] = j

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu >=0表明该query为跟踪query
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1)  # 检测或跟踪与gt的对应关系

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]  # 获取检测query

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]  # 获取跟踪query匹配GT，非新生儿（除了这些之外便是新生儿）

        tgt_state = torch.zeros(len(gt_instances_i), device=pred_logits_i.device)
        tgt_state[tgt_indexes] = 1 # 新生儿为0，跟踪对应的GT为1
        full_tgt_idxes = torch.arange(len(gt_instances_i), device=pred_logits_i.device)
        untracked_tgt_indexes = full_tgt_idxes[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]  # 新生儿的索引

        def match_for_single_decoder_layer(unmatched_outputs, matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.该过程就是DET匈牙利匹配过程
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        # new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes)
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes)

        # step5. update obj_idxes according to the new matching result. 分配GT的ID给track和GT所在的索引
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0) # 当前帧能够匹配到的目标
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # if True:
        #     if len(pred_logits_i):
        #         scores, labels = pred_logits_i.sigmoid().max(-1)
        #         if (labels == 1).any():
        #             print(labels)
        #     if (prev_matched_indices[:, 1]==-1).any():
        #         print(prev_matched_indices)
            
        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs: # 此处匹配时对新生儿时重新算对应关系的，不直接使用最后一层box输出的对应关系
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                if True:
                    new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher, gt_instances_i[full_tgt_idxes], unmatched_track_idxes, full_tgt_idxes)
                else:
                    new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher, gt_instances_i[untracked_tgt_indexes], unmatched_track_idxes, untracked_tgt_indexes)
                # print(new_matched_indices_layer)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})

        if 'ps_outputs' in outputs: # 噪声GT
            for i, aux_outputs in enumerate(outputs['ps_outputs']):
                ar = torch.arange(len(gt_instances_i), device=obj_idxes.device)
                l_dict = self.get_loss('boxes',
                                        aux_outputs,
                                        gt_instances=[gt_instances_i],
                                        indices=[(ar, ar)],
                                        num_boxes=1, )
                self.losses_dict.update(
                    {'frame_{}_ps{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                        l_dict.items()})
        
        # self.calc_loss_for_track_scores(track_instances)
        
        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):  # 实际为一个跟踪ID分配器
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0 # 假如当前帧检测到目标，则disappear_time=0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh) # 挑选新生儿，obj_idxes=-1表示为检测query
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)  # 跟踪中假如置信度偏低则 disappear_time++
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)  # 分配ID
        self.max_obj_id += num_new_objs  # max_obj_id为已有多人ID

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance) # 假如当前帧检测不到，且消失很长时间，则把ID删掉，
        track_instances.obj_idxes[to_del] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        # scores = out_logits[..., 0].sigmoid()
        if len(prob):
            scores, labels = prob.max(-1)
        else:
            scores = out_logits[..., 0].sigmoid()
            labels = torch.full_like(scores, 0, dtype=torch.long)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        mask = (track_instances.labels == -1) | (track_instances.labels == labels)
        track_instances.labels = labels # torch.full_like(scores, 0)
        # track_instances.remove('pred_logits')
        # track_instances.remove('pred_boxes')
        track_instances = track_instances[mask]
        
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False, query_denoise=0):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.query_denoise = query_denoise
        self.position = nn.Embedding(num_queries, 4)
        self.yolox_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]])  # 其实是归一化的box，有这个可以得倒position query
            track_instances.query_pos = torch.cat([self.query_embed.weight, pos2posemb(proposals[:, 4:], d_model) + self.yolox_embed.weight]) # decode的query
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)  # motr decode输出的feature，把这个输入qim中可以获得track的query，某个目标跟踪过程中不再使用query_pos
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)  # ID
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device) # 与匹配到的gt在该图片中的索引
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)  # 消失时间，假如目标跟踪多久后删除该目标
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device) #  与对应GT的IOU
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)   # 实际是当前帧检测输出的置信度
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)  # 检测或跟踪query输出的box框
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)  # 检测或跟踪query输出的置信度argsigmod
        track_instances.labels = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)

        mem_bank_len = self.mem_bank_len  # 默认是不使用的
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)  # 存储历史的output_embedding（经过了一个linear）
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)  # 是否更新了，假如更新为0，不更新为1
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device) # 

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None):
        features, pos = self.backbone(samples)  # 通过backbone获取多次度feature+pad mask+position
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):  # 把backbone输出的feature的通道整成一致，方便concat作为encode输入
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):  # 一般backbone仅输出8/16/32的特征，这个添加一个64的特征
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if gtboxes is not None:  # dn-detr方式训练使用，制作一批噪声GT训练decode
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            # attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            # attn_mask[:n_dt, n_dt:] = True
            # attn_mask[n_dt:, :n_dt] = True
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=torch.float32, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = float('-inf')
            attn_mask[n_dt:, :n_dt] = float('-inf')
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        # transformer 包括encode + decode
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank, mem_bank_pad_mask=track_instances.mem_padding_mask, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):  # 检测头包括box（xc,yc,w,h并归一化1）和class分支
            if lvl == 0:
                reference = init_reference # 实际上就是上一帧跟踪框和decode输入框
            else:
                reference = inter_references[lvl - 1]  # 由于内部是迭代优化的，参考点应为上一次迭代的输出
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}  # 最后一次迭代的输出
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord) # 中间过程迭代的输出
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        if self.query_denoise > 0:  # 由query内有部分噪声GT，因此需要单独拿出来，并放在了ps_outputs下
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)  # 找匹配（跟踪query+检测query）分配GT的ID+ 算loss
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)  # 为存在的目标分配ID，并删除长时间消失的目标ID
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        tmp = {}
        tmp['track_instances'] = track_instances
        if not is_last: # 经过这步后将仅保留有ID的目标，且更新了track的query和pos
            out_track_instances = self.track_embed(tmp)  # 更新跟踪的query，用于下一帧（检测query为学习的，跟踪query则为上一帧输出经过qim变换的特征）
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    # 获取当前帧的跟踪框
    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None): 
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)  # 补pad并或者pad的mask
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals) # 初始化decode的输入或者说目标query，包括（query+pose）
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        # track_instances = self._generate_empty_tracks(proposals)
        res = self._forward_single_image(img,
                                         track_instances=track_instances)  # backbone+encode+decode，获得decode的输出和中间迭代过程输出的box和最后输出的feat(hs,作为经过QIM后可作为下一帧的query)
        res = self._post_process_single_image(res, track_instances, False)  # train是算loss，test时过滤有效跟踪框+为下一帧更新query/pos

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)  # 把box框换算到图像大小
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res: # 把参考点也换算到图像大小
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())
        for frame_index, (frame, gt, proposals) in enumerate(zip(frames, data['gt_instances'], data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            if self.query_denoise > 0:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])
            # track_instances = self._generate_empty_tracks(proposals)

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, gtboxes)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame, gtboxes] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'hs': tmp[2],
                    'aux_outputs': [{
                        'pred_logits': tmp[3+i],
                        'pred_boxes': tmp[3+5+i],
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, gtboxes)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_bdd': 11,
        'e2e_tao': 2000,
        'e2e_bddcc': 100,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
        'e2e_all': 91,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        query_denoise=args.query_denoise,
    )
    model.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    return model, criterion, postprocessors


def img():
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # indx_test = 1
    # train
    image = np.ascontiguousarray(((frames[frame_index].permute(1,2,0).cpu()*torch.tensor([0.229, 0.224, 0.225])+torch.tensor([0.485, 0.456, 0.406]))*255).numpy().astype(np.uint8))
    img_h, img_w, _ = image.shape
    bboxes = self.criterion.gt_instances[frame_index].boxes.cpu().numpy().reshape(-1, 2, 2)
    bboxes[..., 0] *= img_w
    bboxes[..., 1] *= img_h
    bboxes[:, 0] -= bboxes[:, 1]/2
    bboxes[:, 1] += bboxes[:, 0]
    import cv2
    image_copy = image.copy()
    for ith, box in enumerate(bboxes):
        cv2.rectangle(image_copy, pt1 = (int(box[0, 0]), int(box[0, 1])), pt2 =(int(box[1, 0]), int(box[1, 1])), color = (0, 0, 255), thickness = 2)
        cv2.putText(image_copy, '%d'%ith, (int(box[0, 0]), int(box[0, 1])), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    # cv2.imwrite('tmp.jpg', image_copy)
    
    # bboxes = frame_res['aux_outputs'][3]['pred_boxes'].cpu().detach().numpy().reshape(-1, 2, 2)
    bboxes = frame_res['pred_boxes'].cpu().detach().numpy().reshape(-1, 2, 2)
    bboxes[..., 0] *= img_w
    bboxes[..., 1] *= img_h
    bboxes[:, 0] -= bboxes[:, 1]/2
    bboxes[:, 1] += bboxes[:, 0]
    scores = frame_res['pred_logits'].cpu().sigmoid().detach().numpy().reshape(len(bboxes), -1).max(-1)
    classes = frame_res['pred_logits'].cpu().sigmoid().detach().numpy().reshape(len(bboxes), -1).argmax(-1)
    for ith, (box, s, c) in enumerate(zip(bboxes, scores, classes)):
        if s < 0.5: continue
        cv2.rectangle(image_copy, pt1 = (int(box[0, 0]), int(box[0, 1])), pt2 =(int(box[1, 0]), int(box[1, 1])), color = (0, 255, 0), thickness = 2)
        cv2.putText(image_copy, '%.2f_%d'%(s, c), (int(box[0, 0]), int(box[0, 1])), 0, 0.3, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite('tmp.jpg', image_copy)
    
    # test
    image = np.ascontiguousarray(((img.tensors[0].permute(1,2,0).cpu()*torch.tensor([0.229, 0.224, 0.225])+torch.tensor([0.485, 0.456, 0.406]))*255).numpy().astype(np.uint8))
    img_h, img_w, _ = image.shape
    bboxes = track_instances.ref_pts.cpu().numpy().reshape(-1, 2, 2)
    bboxes[..., 0] *= img_w
    bboxes[..., 1] *= img_h
    bboxes[:, 0] -= bboxes[:, 1]/2
    bboxes[:, 1] += bboxes[:, 0]
    import cv2
    for i in range(68):
        image_copy = image.copy()
        for box in bboxes[5*i:5*(i+1)]:
            cv2.rectangle(image_copy, pt1 = (int(box[0, 0]), int(box[0, 1])), pt2 =(int(box[1, 0]), int(box[1, 1])), color = (0, 0, 255), thickness = 2)
        cv2.imwrite('tmp2/%d.jpg'%i, image_copy)