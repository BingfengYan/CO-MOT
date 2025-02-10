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
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances

from random import choice, randint


def is_crowd(ann):
    return 'extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1

attr_dict = dict()
attr_dict["categories"] = [
    {"supercategory": "none", "id": 0, "name": "pedestrian"},
    {"supercategory": "none", "id": 1, "name": "rider"},
    {"supercategory": "none", "id": 2, "name": "car"},
    {"supercategory": "none", "id": 3, "name": "truck"},
    {"supercategory": "none", "id": 4, "name": "bus"},
    {"supercategory": "none", "id": 5, "name": "train"},
    {"supercategory": "none", "id": 6, "name": "motorcycle"},
    {"supercategory": "none", "id": 7, "name": "bicycle"},
    {"supercategory": "none", "id": 8, "name": "other person"},
    {"supercategory": "none", "id": 9, "name": "trailer"},
    {"supercategory": "none", "id": 10, "name": "other vehicle"}
]
attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path

        self.labels_full = defaultdict(lambda : defaultdict(list))
        self.images_full = defaultdict(lambda : defaultdict())
        self._add_mot_folder(data_txt_path.replace('images/track', 'labels/box_track_20'), data_txt_path)
        self._add_mot_folder('BDD100K/images/track/val/'.replace('images/track', 'labels/box_track_20'), 'BDD100K/images/track/val/')
        
        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"Found {len(vid_files)} videos, {len(self.indices)} frames")

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

        # crowdhuman
        self.ch_dir = Path(args.mot_path) / 'crowdhuman'
        self.ch_indices = []
        if args.append_crowd2:
            for line in open(self.ch_dir / f"annotation_trainval.odgt"):
                datum = json.loads(line)
                boxes = [ann['fbox'] for ann in datum['gtboxes'] if not is_crowd(ann)]
                self.ch_indices.append((datum['ID'], boxes))
        # self.ch_indices = self.ch_indices + self.ch_indices
        print(f"Found {len(self.ch_indices)} images")

        if args.det_db:
            with open(os.path.join(args.mot_path, args.det_db)) as f:
                self.det_db = json.load(f)
        else:
            self.det_db = defaultdict(list)

    def _add_mot_folder(self, split_dir, video_dir):
        print("Adding", split_dir, video_dir)
        for label_json in os.listdir(os.path.join(self.mot_path, split_dir)):
            with open(os.path.join(self.mot_path, split_dir, label_json)) as f:
                labels_json = json.load(f)
                for label_json in labels_json:
                    img_name = label_json['name']
                    video_name = os.path.join(video_dir,label_json['videoName'])
                    labels = label_json['labels']
                    t = label_json['frameIndex']
                    self.images_full[video_name][t] = img_name
                    for label in labels:
                        category = label['category']
                        x1 = label['box2d']['x1']
                        x2 = label['box2d']['x2']
                        y1 = label['box2d']['y1']
                        y2 = label['box2d']['y2']
                        width = x2 - x1
                        height = y2 - y1
                        identity = int(label['id'])
                        crowd = False
                        # [class] [identity] [x_center] [y_center] [width] [height]
                        # txt_string += "{} {} {} {} {} {}\n".format(attr_id_dict[category], identity, x_center, y_center, width, height)
                        self.labels_full[video_name][t].append([x1, y1, width, height, identity, crowd, attr_id_dict[category]])

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def load_crowd(self, index):
        ID, boxes = self.ch_indices[index]
        boxes = copy.deepcopy(boxes)
        img_path = self.ch_dir / 'Crowdhuman_train' / f'{ID}.jpg'
        if not os.path.exists(img_path): img_path = self.ch_dir / 'Crowdhuman_val' / f'{ID}.jpg'
        img = Image.open(img_path)

        w, h = img._size
        n_gts = len(boxes)
        scores = [0. for _ in range(len(boxes))]
        if f'crowdhuman/train_image/{ID}.txt' in self.det_db:
            for line in self.det_db[f'crowdhuman/train_image/{ID}.txt']:
                *box, s = map(float, line.split(','))
                boxes.append(box)
                scores.append(s)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            'boxes': boxes,
            'scores': torch.as_tensor(scores),
            'labels': torch.zeros((n_gts, ), dtype=torch.long),
            'iscrowd': torch.zeros((n_gts, ), dtype=torch.bool),
            'image_id': torch.tensor([0]),
            'area': areas,
            'obj_ids': torch.arange(n_gts),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w]),
            'dataset': "CrowdHuman",
        }
        rs = T.FixedMotRandomShift(self.num_frames_per_batch)
        return rs([img], [target])

    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.mot_path, vid, self.images_full[vid][idx])
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.

        targets['dataset'] = 'MOT17'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for *xywh, id, crowd, cl in self.labels_full[vid][idx]:
            targets['boxes'].append(xywh)
            assert not crowd
            targets['iscrowd'].append(crowd)
            targets['labels'].append(int(cl))
            targets['obj_ids'].append(id + obj_idx_offset)
            targets['scores'].append(1.)
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        if txt_key.replace('dancetrack/', 'DanceTrack/') in self.det_db:
            for line in self.det_db[txt_key.replace('dancetrack/', 'DanceTrack/')]:
                *box, s = map(float, line.split(','))
                targets['boxes'].append(box)
                targets['scores'].append(s)

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.long)
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        if idx < len(self.indices):
            vid, f_index = self.indices[idx]
            indices = self.sample_indices(vid, f_index)
            images, targets = self.pre_continuous_frames(vid, indices)
        else:
            images, targets = self.load_crowd(idx - len(self.indices))
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,
        }

    def __len__(self):
        return len(self.indices) + len(self.ch_indices)


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        super().__init__(args, data_txt_path, seqs_folder, transform)

    def __getitem__(self, idx):
        
        vid = list(self.vid_tmax.keys())[idx]
        t_min = min(self.labels_full[vid].keys())
        t_max = max(self.labels_full[vid].keys()) + 1
        # indices = range(t_min, t_max)
        # images, targets = self.pre_continuous_frames(vid, indices)

        # if self.transform is not None:
        #     images, targets = self.transform(images, targets)
        # gt_instances, proposals = [], []
        # for img_i, targets_i in zip(images, targets):
        #     gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
        #     gt_instances.append(gt_instances_i)
        #     n_gt = len(targets_i['labels'])
        #     proposals.append(torch.cat([
        #         targets_i['boxes'][n_gt:],
        #         targets_i['scores'][n_gt:, None],
        #     ], dim=1))
        # return {
        #     'imgs': images,
        #     'gt_instances': gt_instances,
        #     'proposals': proposals,
        # }
        return {
            'video_name': os.path.join(self.mot_path, vid),
            'video_min': t_min,
            'video_max': t_max,
        }

    def __len__(self):
        return len(self.vid_tmax)


def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    if image_set == 'train':
        return mot17_train
    elif image_set == 'val':
        return mot17_test
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transform = build_transform(args, image_set)
    if image_set == 'train':
        data_txt_path = "BDD100K/images/track/train/"
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, transform=transform)
    if image_set == 'val':
        data_txt_path = "BDD100K/images/track/val/"
        dataset = DetMOTDetectionValidation(args, data_txt_path=data_txt_path, seqs_folder=root, transform=transform)
    return dataset
