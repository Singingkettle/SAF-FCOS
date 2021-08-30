from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from numpy import asarray
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


def calculate_mean_std_of_img(img):
    img = asarray(img)
    # convert from integers to floats
    img = img.astype('float64')
    means = img.mean(axis=(0, 1), dtype='float64')
    stds = img.std(axis=(0, 1), dtype='float64')
    return means, stds


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
    json_name = 'gt_nuscenes_coco_%s.json'
    for data_set, set_annos in zip(sets, all_annos):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        num_item = 0
        norm_param = {}
        im_means = None
        im_stds = None
        pc_im_means = None
        pc_im_stds = None
        for annos in set_annos:
            if len(annos) > 0:
                num_item += 1
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
                pointsensor_token = sample['data']['RADAR_FRONT']
                pc_rec = nusc.get('sample_data', pointsensor_token)

                image['file_name'] = sample_data['filename']
                image['pc_file_name'] = pc_rec['filename'].replace('samples', 'imagepc').replace('pcd', 'jpg')
                images.append(image)

                if num_item == 1:
                    img_data = Image.open(os.path.join(data_dir, image['file_name']))
                    pc_img_data = Image.open(os.path.join(data_dir, image['pc_file_name']))
                    im_means, im_stds = calculate_mean_std_of_img(img_data)
                    pc_im_means, pc_im_stds = calculate_mean_std_of_img(pc_img_data)
                else:
                    img_data = Image.open(os.path.join(data_dir, image['file_name']))
                    pc_img_data = Image.open(os.path.join(data_dir, image['pc_file_name']))
                    tmp_im_means, tmp_im_stds = calculate_mean_std_of_img(img_data)
                    tmp_pc_im_means, tmp_pc_im_stds = calculate_mean_std_of_img(pc_img_data)

                    im_means = (num_item - 1) / num_item * im_means + tmp_im_means / num_item
                    im_stds = (num_item - 1) / num_item * im_stds + tmp_im_stds / num_item
                    pc_im_means = (num_item - 1) / num_item * pc_im_means + tmp_pc_im_means / num_item
                    pc_im_stds = (num_item - 1) / num_item * pc_im_stds + tmp_pc_im_stds / num_item

                pc = np.loadtxt(os.path.join(data_dir, pc_rec['filename'].replace('samples', 'pc')))
                if len(pc.shape) == 1:
                    if pc.shape[0] == 0:
                        continue
                    else:
                        pc = np.expand_dims(pc, axis=1)

                for anno in annos:
                    ann = dict()
                    ann['id'] = ann_id
                    ann_id += 1
                    ann['image_id'] = image['id']
                    ann['category_id'] = 1
                    ann['iscrowd'] = 0
                    xyxy_box = anno['bbox_corners']
                    legal_box = False
                    for pc_index in range(pc.shape[1]):
                        point = pc[:, pc_index]
                        if (point[0] > xyxy_box[0]) and (point[0] < xyxy_box[1]) and (
                                point[1] > xyxy_box[1]) and (point[1] < xyxy_box[3]):
                            legal_box = True
                            break
                    ann['legal'] = legal_box
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

        norm_param['im_means'] = [item for item in im_means]
        norm_param['im_stds'] = [item for item in im_stds]
        norm_param['pc_im_means'] = [item for item in pc_im_means]
        norm_param['pc_im_stds'] = [item for item in pc_im_stds]
        with open(os.path.join(out_dir, 'norm_param' + json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(norm_param))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "nuscenes":
        merge_convert_and_detect(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
