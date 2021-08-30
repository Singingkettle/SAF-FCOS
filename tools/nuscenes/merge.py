#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# This file is copy from https://github.com/facebookresearch/Detectron/tree/master/tools

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--dataset', help="convert dataset to coco-style", default='nuscenes', type=str)
    parser.add_argument('--datadir', help="data dir for annotations to be converted",
                        default='/home/citybuster/Data/nuScenes', type=str)
    parser.add_argument('--outdir', help="output dir for json files",
                        default='/home/citybuster/Data/nuScenes/v1.0-trainval', type=str)

    return parser.parse_args()


def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box


def xyxy_to_polygn(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    xywh_box = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
    return xywh_box


def merge_convert_and_detect(data_dir, out_dir):
    """merge detection results and convert results to COCO"""

    # Load mini dataset, and dataset, the mini dataset is used as test and valid dataset
    nusc_mini = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=True)
    sample_files_mini = [s['filename'] for s in nusc_mini.sample_data if (s['channel'] == 'CAM_FRONT') and
                         s['is_key_frame']]
    sample_files = [s['filename'] for s in nusc.sample_data if (s['channel'] == 'CAM_FRONT') and
                    s['is_key_frame']]
    sample_files_mini = set(sample_files_mini)
    sample_files = set(sample_files)

    # Filter dataset items, ensure the mini dataset and dataset are not cross
    tmp_sample_files = []
    for item in sample_files:
        if item not in sample_files_mini:
            tmp_sample_files.append(item)
        else:
            continue

    train_sample_files = tmp_sample_files
    tv_sample_files = sample_files_mini

    with open(os.path.join(data_dir, 'v1.0-trainval', 'image_pc_annotations.json'), 'r') as f:
        sample_labels = json.load(f)

    train_annos = list()
    tv_annos = list()

    for i, annos in tqdm(enumerate(sample_labels)):
        if len(annos) > 0:
            sample_data_token = annos[0]['sample_data_token']
            sample_data = nusc.get('sample_data', sample_data_token)
            im_file_name = sample_data['filename']
            if im_file_name in train_sample_files:
                train_annos.append(annos)
            elif im_file_name in tv_sample_files:
                tv_annos.append(annos)
            else:
                sys.exit("Something wrong with the generate json labels, Please check your code carefully.")

    sets = ['train', 'testval']
    all_annos = [train_annos, tv_annos]
    img_id = 0
    ann_id = 0
    json_name = 'coco_%s.json'
    for data_set, set_annos in zip(sets, all_annos):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        for annos in set_annos:
            if len(annos) > 0:
                if len(images) % 50 == 0:
                    print("Processed %s images, %s annotations" % (
                        len(images), len(annotations)))
                image = dict()
                image['id'] = img_id
                img_id += 1

                image['width'] = 1600
                image['height'] = 900

                sample_data_token = annos[0]['sample_data_token']
                sample_data = nusc.get('sample_data', sample_data_token)
                sample = nusc.get('sample', sample_data['sample_token'])
                scene_token = sample['scene_token']
                scene = nusc.get('scene', scene_token)
                scene_description = scene['description']
                pointsensor_token = sample['data']['RADAR_FRONT']
                pc_rec = nusc.get('sample_data', pointsensor_token)

                image['file_name'] = sample_data['filename']
                image['pc_file_name'] = pc_rec['filename'].replace('samples', 'imagepc').replace('pcd', 'jpg')
                images.append(image)

                if scene_description.find('Night') == -1:
                    # Load detection results
                    with open(
                            os.path.join(data_dir, image['file_name'].replace('samples', 'fcos_nuscenes').replace('jpg', 'txt')),
                            'r') as f:
                        detection_list = f.readlines()
                        if len(detection_list) > 0:
                            pc = np.loadtxt(os.path.join(data_dir, pc_rec['filename'].replace('samples', 'pc')))
                            if len(pc.shape) == 1:
                                if pc.shape[0] == 0:
                                    continue
                                else:
                                    pc = np.expand_dims(pc, axis=1)

                            for line_str in detection_list:
                                line_str = line_str.strip('\n').strip('\t')
                                line_str = line_str.split(',')
                                line_gt = [float(x) for x in line_str]
                                xyxy_box = line_gt[1:]
                                legal_box = False
                                for pc_index in range(pc.shape[1]):
                                    point = pc[:, pc_index]
                                    if (point[0] > xyxy_box[0]) and (point[0] < xyxy_box[1]) and (
                                            point[1] > xyxy_box[1]) and (point[1] < xyxy_box[3]):
                                        legal_box = True
                                        break
                                if legal_box:
                                    ann = dict()
                                    ann['id'] = ann_id
                                    ann_id += 1
                                    ann['image_id'] = image['id']
                                    ann['category_id'] = 1
                                    ann['iscrowd'] = 0
                                    xywh_box = xyxy_to_xywh(xyxy_box)
                                    ann['bbox'] = xywh_box
                                    ann['area'] = xywh_box[2] * xywh_box[3]
                                    ann['segmentation'] = xyxy_to_polygn(xyxy_box)
                                    annotations.append(ann)
                else:
                    for anno in annos:
                        ann = dict()
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']
                        ann['category_id'] = 1
                        ann['iscrowd'] = 0
                        xyxy_box = anno['bbox_corners']
                        xywh_box = xyxy_to_xywh(xyxy_box)
                        ann['bbox'] = xywh_box
                        ann['area'] = xywh_box[2] * xywh_box[3]
                        ann['segmentation'] = xyxy_to_polygn(xyxy_box)
                        annotations.append(ann)
            else:
                continue

        categories = [{"id": 1, "name": 'vehicle'}]
        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "nuscenes":
        merge_convert_and_detect(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
