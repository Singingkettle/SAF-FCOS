import argparse
import json
import os
import sys
from concurrent import futures

import cv2
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes

_DISTANCE_RANGE = [0, 250]
_SPEED_RANGE = [-33, 33]


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def draw_pc_image(pc, save_path, radius, im_height=900, im_width=1600):
    img_b = np.zeros((im_height, im_width), np.uint8)
    img_g = np.zeros((im_height, im_width), np.uint8)
    img_r = np.zeros((im_height, im_width), np.uint8)
    his_mask = np.zeros((im_height, im_width), np.uint8)
    pc = pc[:, pc[2, :].argsort()]
    num_points = pc.shape[1]
    # Line thickness of 2 px
    thickness = -1

    point_id = 0
    for i in range(num_points):
        center_coordinates = (int(pc[0, i]), int(pc[1, i]))
        depth = pc[2, i]
        vx = pc[6, i]
        vy = pc[7, i]
        if (depth > _DISTANCE_RANGE[0]) and (depth < _DISTANCE_RANGE[1]):
            v = np.sqrt(vx ** 2 + vy ** 2)
            if (v > _SPEED_RANGE[0]) and (v < _SPEED_RANGE[1]):
                point_id += 1
                red = int(depth / 250 * 128 + 127)
                green = int((vx + 20) / 40 * 128 + 127)
                blue = int((vy + 20) / 40 * 128 + 127)
                color = (blue, green, red)
                color = np.asarray(color).astype(np.uint8)
                cur_mask = np.zeros((900, 1600), np.uint8)
                cur_mask = cv2.circle(cur_mask, center_coordinates, radius, 1, thickness)
                save_cur_mask = cur_mask - cur_mask * his_mask
                img_b = img_b + save_cur_mask * color[2]
                img_g = img_g + save_cur_mask * color[1]
                img_r = img_r + save_cur_mask * color[0]
                his_mask = his_mask + save_cur_mask

    if point_id > 0:
        im = np.stack([img_r, img_g, img_b], axis=2)
        image = Image.fromarray(im, 'RGB')
        image.save(save_path)
        norm_info = {}

        norm_save_path = save_path.replace('.png', '.json')
        # convert from integers to floats
        im = im.astype('float64')
        means = im.mean(axis=(0, 1), dtype='float64')
        stds = im.std(axis=(0, 1), dtype='float64')
        mean = np.reshape(means, [3, 1])
        std = np.reshape(stds, [3, 1])
        norm_info['mean'] = (mean[0, 0], mean[1, 0], mean[2, 0])
        norm_info['std'] = (std[0, 0], std[1, 0], std[2, 0])
        with open(norm_save_path, 'w') as f:
            json.dump(norm_info, f, sort_keys=True, indent=4)


def convert_pcd_file(sample_data_token: str, radius: float):
    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    point_sensor_token = s_rec['data']['RADAR_FRONT']
    pc_rec = nusc.get('sample_data', point_sensor_token)
    pcd_path = os.path.join(nusc.dataroot, pc_rec['filename'].replace('samples', 'pc'))
    save_path = os.path.join(nusc.dataroot,
                             pc_rec['filename'].replace('samples', 'imagepc_%02d' % radius).replace('pcd', 'png'))
    save_folder = os.path.dirname(save_path)
    if os.path.isfile(save_path):
        return
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    pc = np.loadtxt(pcd_path)
    if len(pc.shape) == 1:
        if pc.shape[0] > 0:
            pc = np.expand_dims(pc, axis=1)
            draw_pc_image(pc, save_path, radius)
    else:
        draw_pc_image(pc, save_path, radius)


def main(args):
    """Generates 2D re-projections of the 3D bounding boxes present in the dataset."""

    # Get tokens for all camera images.
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['channel'] == 'CAM_FRONT') and
                                 s['is_key_frame']]

    # Make dirs
    if not os.path.exists(os.path.join(nusc.dataroot, 'pc')):
        os.makedirs(os.path.join(nusc.dataroot, 'pc', 'RADAR_FRONT'))

    # Convert pcd file to image file
    print("Generating 2D radar image")
    radius_list = [1, 3, 5, 7, 9, 11]
    num_threads = 40
    num_tokens = len(sample_data_camera_tokens)
    for radius in radius_list:
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(convert_pcd_file, token, radius) for token in
                  sample_data_camera_tokens]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, num_tokens, prefix=args.version, suffix='Done ', barLength=40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/home/citybuster/Data/nuScenes/',
                        help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version.')
    parser.add_argument('--filename', type=str, default='image_pc_annotations.json', help='Output filename.')
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main(args)
