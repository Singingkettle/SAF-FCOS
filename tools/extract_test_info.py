# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from glob import glob

import torch
from fcos_core.config import cfg
from fcos_core.data.build import build_dataset
from fcos_core.data.transforms import build_transforms
from fcos_core.utils.imports import import_file
from fcos_core.data.datasets.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/home/citybuster/Projects/FCOS/configs/fcos_nuscenes/",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/citybuster/Projects/FCOS/training_dir/",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()

    checkpointer_list = glob(os.path.join(cfg.OUTPUT_DIR, 'model_0*.pth'))
    paths_catalog = import_file(
        "fcos_core.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    is_train = False
    box_only = False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY
    expected_results = cfg.TEST.EXPECTED_RESULTS
    expected_results_sigma_tol = cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    checkpointer_list.sort()

    for checkpointer_file in checkpointer_list:
        model_name = checkpointer_file.split('/')[-1]
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference-" + model_name, dataset_name)
                output_folders[idx] = output_folder

        # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
        transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
        datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)
        for output_folder, dataset_name, dataset in zip(output_folders, dataset_names, datasets):
            predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
            extra_args = dict(
                box_only=box_only,
                iou_types=iou_types,
                expected_results=expected_results,
                expected_results_sigma_tol=expected_results_sigma_tol,
            )

            evaluate(dataset=dataset,
                     predictions=predictions,
                     output_folder=output_folder,
                     **extra_args)


if __name__ == "__main__":
    main()
