# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from fcos_core.modeling.rpn.fcos.loss import make_fcos_loss_evaluator
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
# INF = 100000000
#
# cfg.merge_from_file('/home/citybuster/Projects/FCOS/configs/fcos_nuscenes/fcos_imprv_R_50_FPN_1x_ADD.yaml')
# cfg.freeze()
#
# loss_evaluator = make_fcos_loss_evaluator(cfg)
# loss_evaluator(locations, box_cls, box_regression, centerness, targets)
#
# # model = build_detection_model(cfg)
# # device = torch.device(cfg.MODEL.DEVICE)
# # model.to(device)
# # optimizer = make_optimizer(cfg, model)
# # scheduler = make_lr_scheduler(cfg, optimizer)
# # checkpointer = DetectronCheckpointer(
# #         cfg, model, optimizer, scheduler, './', False
# #     )
# # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
# data_loader = make_data_loader(
#     cfg,
#     is_train=True,
#     is_distributed=False,
#     start_iter=1,
# )
#
# for iteration, (images, pc_images, targets, _) in enumerate(data_loader, 1):
#     tmp = 1

import cv2
import numpy as np
import copy
from PIL import Image

im1 = cv2.imread('/home/citybuster/Data/nuScenes/imagepc_10/RADAR_FRONT/n008-2018-05-21-11-06-59-0400__RADAR_FRONT__1526915243042374.jpg')
im2 = cv2.imread('/home/citybuster/Data/nuScenes/imagepc/RADAR_FRONT/n008-2018-05-21-11-06-59-0400__RADAR_FRONT__1526915243042374.jpg')
pc = np.loadtxt('/home/citybuster/Data/nuScenes/pc/RADAR_FRONT/n008-2018-05-21-11-06-59-0400__RADAR_FRONT__1526915243042374.pcd')
pc_sorted = pc[:, pc[2,:].argsort()]

img = np.zeros((900, 1600, 3), np.uint8)
num_points = pc.shape[1]
# Line thickness of 2 px
thickness = -1
radius = 10

point_id = 0
for i in range(num_points):
    center_coordinates = (int(pc[0, i]), int(pc[1, i]))
    depth = pc[2, i]
    vx = pc[6, i]
    vy = pc[7, i]
    point_id += 1
    red = int(depth / 250 * 128 + 127)
    green = int((vx + 20) / 40 * 128 + 127)
    blue = int((vy + 20) / 40 * 128 + 127)
    color = (blue, green, red)
    img = cv2.circle(img, center_coordinates, radius, color, thickness)

# np.savetxt('imb.txt', img[:, :, 0], fmt='%d')
# np.savetxt('img.txt', img[:, :, 1], fmt='%d')
# np.savetxt('imr.txt', img[:, :, 2], fmt='%d')

img_b = np.zeros((900, 1600), np.uint8)
img_g = np.zeros((900, 1600), np.uint8)
img_r = np.zeros((900, 1600), np.uint8)
num_points = pc.shape[1]
# Line thickness of 2 px
thickness = -1
radius = 10

point_id = 0
his_mask = np.zeros((900, 1600), np.uint8)
for i in range(num_points):
    if i == num_points-1:
        tmp = 1
    center_coordinates = (int(pc[0, i]), int(pc[1, i]))
    depth = pc[2, i]
    vx = pc[6, i]
    vy = pc[7, i]
    point_id += 1
    red = int(depth / 250 * 128 + 127)
    green = int((vx + 20) / 40 * 128 + 127)
    blue = int((vy + 20) / 40 * 128 + 127)
    color = (red, green, blue)
    color = np.asarray(color).astype(np.uint8)
    print(color)
    cur_mask = np.zeros((900, 1600), np.uint8)
    cur_mask = cv2.circle(cur_mask, center_coordinates, radius, 1, thickness)
    # print(cur_mask.sum())
    save_cur_mask = cur_mask - cur_mask * his_mask
    img_b = img_b + save_cur_mask * color[2]
    img_g = img_g + save_cur_mask * color[1]
    img_r = img_r + save_cur_mask * color[0]
    his_mask = his_mask + save_cur_mask

im = np.stack([img_r, img_g, img_b], axis=2)
tmp = im[:, :, 0]
print(tmp.max())
tmp = im[:, :, 1]
print(tmp.max())
tmp = im[:, :, 2]
print(tmp.max())
image = Image.fromarray(im, 'RGB')
image.save('test.png')
# cv2.imwrite('test.jpg', im)
np.savetxt('imb.txt', img_b, fmt='%d')
np.savetxt('img.txt', img_g, fmt='%d')
np.savetxt('imr.txt', img_r, fmt='%d')

print('=========================')
im = cv2.imread('test.png')
tmp = im[:, :, 0]
print(tmp.max())
tmp = im[:, :, 1]
print(tmp.max())
tmp = im[:, :, 2]
print(tmp.max())
print('===========================')
tmp = im - np.stack([img_b, img_g, img_r], axis=2)
print(tmp.max())
print(tmp.min())
