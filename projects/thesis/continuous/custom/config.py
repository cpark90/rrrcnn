# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_custom_config(cfg):
    """
    Add custom configuration.
    """
    _C = cfg

    _C.MODEL.CUSTOM = CN()
    _C.MODEL.CUSTOM.CLASSES = 10
    _C.MODEL.CUSTOM.INPUT_SIZE = 224
    _C.MODEL.CUSTOM.CHANNELS = 20
    _C.MODEL.CUSTOM.NUM_KERNEL = 5
    _C.MODEL.CUSTOM.FOCAL_LOSS_ALPHA = 2.0
    _C.MODEL.CUSTOM.FOCAL_LOSS_GAMMA = 0.25
    _C.MODEL.CUSTOM.NORM = 'BN'
    _C.MODEL.CUSTOM.PSR = CN()
    _C.MODEL.CUSTOM.PSR.CONVF_NAME = ""
    _C.MODEL.CUSTOM.PSR.CONV_NAME = ""
    _C.MODEL.CUSTOM.PSR.ROT_1x1 = False
    _C.MODEL.CUSTOM.PSR.NOISE_VAR = 0.0
    _C.MODEL.CUSTOM.PS = CN()
    _C.MODEL.CUSTOM.PS.KERNEL_TYPE = 1
    _C.MODEL.CUSTOM.PS.SPATIAL = False
    _C.MODEL.CUSTOM.PS.SPATIAL_1x1 = False
    _C.MODEL.CUSTOM.PS.NOISE_VAR = 0.0
    _C.MODEL.CUSTOM.RESNETS = CN()
    _C.MODEL.CUSTOM.RESNETS.ROT_1x1 = False
    _C.MODEL.CUSTOM.RESNETS.NOISE_VAR = 0.0
    _C.MODEL.CUSTOM.RESNETS.STEM = CN()
    _C.MODEL.CUSTOM.RESNETS.STEM.CONVF_7x7 = True
    _C.MODEL.CUSTOM.RESNETS.STEM.CONVF_NAME = ""
    _C.MODEL.CUSTOM.RESNETS.STEM.STRIDE_PSR = 1
    _C.MODEL.CUSTOM.RESNETS.BLOCK = CN()
    _C.MODEL.CUSTOM.RESNETS.BLOCK.CONV_NAME = ""
    _C.MODEL.CUSTOM.RESNETS.BLOCK.CONV_1x1_ROT = False
    _C.MODEL.CUSTOM.FPN = CN()
    _C.MODEL.CUSTOM.FPN.CONVF_NAME = ""
    _C.MODEL.CUSTOM.FPN.CONV_NAME = ""
    _C.MODEL.CUSTOM.FPN.NOISE_VAR = 0.0
    _C.MODEL.CUSTOM.BRANCH = CN()
    _C.MODEL.CUSTOM.BRANCH.NUM_BRANCH = 1


    _C.DATASETS.CUSTOM = CN()
    _C.DATASETS.CUSTOM.DATASET_MAPPER = "GeneralDatasetMapper"
    _C.DATASETS.CUSTOM.ROTATION_TRAIN = 0
    _C.DATASETS.CUSTOM.ROTATION_TEST = 0

