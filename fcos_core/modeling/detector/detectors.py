# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This code is re-writen by shuo chang for detection based on mmradar and visual data.
"""
from .generalized_rcnn import GeneralizedRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
