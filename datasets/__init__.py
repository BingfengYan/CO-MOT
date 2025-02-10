# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .dance import build as build_e2e_dance
from .dance_test import build as build_e2e_dance_test
from .tao import build as build_e2e_tao
from .joint import build as build_e2e_joint
from .mot import build as build_e2e_mot
from .all import build as build_e2e_all
from .bdd100k import build as build_e2e_bdd
from .bdd100kcoco import build as build_e2e_bddcc


def build_dataset(image_set, args):
    if args.dataset_file == 'e2e_joint':
        return build_e2e_joint(image_set, args)
    elif args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)
    elif args.dataset_file == 'e2e_dance_test':
        return build_e2e_dance_test(image_set, args)
    elif args.dataset_file == 'e2e_all':
        return build_e2e_all(image_set, args)
    elif args.dataset_file == 'e2e_bdd':
        return build_e2e_bdd(image_set, args)
    elif args.dataset_file == 'e2e_tao':
        return build_e2e_tao(image_set, args)
    elif args.dataset_file == 'e2e_bddcc':
        return build_e2e_bddcc(image_set, args)
    elif args.dataset_file == 'e2e_mot':
        return build_e2e_mot(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
