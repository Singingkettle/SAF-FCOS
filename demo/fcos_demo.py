# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import json
import os
from shutil import copyfile

import fcos_core.config.globalvar as gl
from tqdm import tqdm

from fcos_core.config import cfg
from predictor import NuScenesDemo


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="/home/citybuster/Projects/FCOS/configs/fcos_nuscenes/fcos_imprv_R_50_FPN_1x_ATTMIX_135_Circle_07.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="/home/citybuster/Projects/FCOS/training_dir/fcos_imprv_R_50_FPN_1x_ATTMIX_135_Circle_07"
                "/model_0037500.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = [0.5335188508033752]

    with open(os.path.join('/home/citybuster/Data/nuScenes/v1.0-trainval/gt_fcos_coco_test_v2.json'), 'r') as f:
        images = json.load(f)['images']
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = NuScenesDemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )
    gl._init()
    for image in tqdm(images):
        img_path = os.path.join('/home/citybuster/Data/nuScenes', image['file_name'])
        pc_img_path = os.path.join('/home/citybuster/Data/nuScenes',
                                   image['pc_file_name'].replace('imagepc', 'imagepc_07'))
        attention_map_path = os.path.join('/home/citybuster/Data/nuScenes',
                                          image['file_name'].replace('samples', 'attention_map'))
        copyfile(img_path, attention_map_path)
        # attention_map_path = attention_map_path.replace('jpg', 'txt')

        # gl.set_value('path', attention_map_path)
        # save_folder = os.path.dirname(attention_map_path)
        # if not os.path.isdir(save_folder):
        #     os.makedirs(save_folder)
        # img = cv2.imread(img_path)
        # pc_img = cv2.imread(pc_img_path)
        # if img is None:
        #     continue
        # start_time = time.time()
        # composite = coco_demo.run_on_opencv_image(img, pc_img)
        # print("{}\tinference time: {:.2f}s".format(image['file_name'], time.time() - start_time))
        # cv2.imshow(image['file_name'], composite)
    # print("Press any keys to exit ...")
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
