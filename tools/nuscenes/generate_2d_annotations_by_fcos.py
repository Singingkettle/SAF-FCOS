from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os

import numpy as np
from nuscenes.nuscenes import NuScenes


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


def run(data_dir, out_dir):
    """merge detection results and convert results to COCO"""

    # Load mini dataset, test dataset and dataset, the mini dataset is used as valid dataset
    nusc_mini = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    nusc_test = NuScenes(version='v1.0-test', dataroot=data_dir, verbose=True)
    nusc_train = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=True)
    mini_sample_tokens = [s['token'] for s in nusc_mini.sample_data if (s['channel'] == 'CAM_FRONT') and
                          s['is_key_frame']]
    test_sample_tokens = [s['token'] for s in nusc_test.sample_data if (s['channel'] == 'CAM_FRONT') and
                          s['is_key_frame']]
    sample_tokens = [s['token'] for s in nusc_train.sample_data if (s['channel'] == 'CAM_FRONT') and
                     s['is_key_frame']]

    # Filter dataset items, ensure the mini dataset and dataset are not cross
    tmp_sample_tokens = []
    for item in sample_tokens:
        if item not in mini_sample_tokens:
            tmp_sample_tokens.append(item)
        else:
            continue

    train_sample_tokens = tmp_sample_tokens

    sets = ['train', 'val', 'test']
    all_tokens = [train_sample_tokens, mini_sample_tokens, test_sample_tokens]
    all_nuscs = {'train': nusc_train, 'val': nusc_mini, 'test': nusc_test}
    img_id = 0
    ann_id = 0
    json_name = 'gt_fcos_coco_%s.json'
    for data_set, token_list in zip(sets, all_tokens):
        nusc = all_nuscs[data_set]
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        for sample_data_token in token_list:
            # if len(images) % 50 == 0:
            #     print("Processed %s images, %s annotations" % (
            #         len(images), len(annotations)))

            try:
                sample_data = nusc.get('sample_data', sample_data_token)
                sample = nusc.get('sample', sample_data['sample_token'])
                point_sensor_token = sample['data']['RADAR_FRONT']
                pc_rec = nusc.get('sample_data', point_sensor_token)
            except:
                print(sample_data_token)
                sample_data = None
                point_sensor_token = None
                pc_rec = None
            if point_sensor_token is None:
                continue

            image = dict()
            image['id'] = img_id
            img_id += 1

            image['width'] = 1600
            image['height'] = 900
            image['file_name'] = sample_data['filename']
            image['pc_file_name'] = pc_rec['filename'].replace('samples', 'imagepc').replace('pcd', 'png')

            pc_image_path = os.path.join(data_dir, image['pc_file_name'].replace('imagepc', 'imagepc_01'))
            if not os.path.isfile(pc_image_path):
                print(pc_image_path)
            images.append(image)

            with open(os.path.join(data_dir, image['file_name'].replace('samples', 'fcos').replace('jpg', 'txt')),
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
                            if (point[0] > xyxy_box[0]) and (point[0] < xyxy_box[2]) and (
                                    point[1] > xyxy_box[1]) and (point[1] < xyxy_box[3]):
                                legal_box = True
                                break
                        ann = dict()
                        ann['legal'] = legal_box
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
        run(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
