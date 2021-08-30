# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from PIL import Image
import os
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints
from . import coco
from fcos_core.config import cfg


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class NuScenesDataset(coco.COCODataset):
    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None):
        super(NuScenesDataset, self).__init__(ann_file, root, remove_images_without_annotations, transforms)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        im_path = coco.loadImgs(img_id)[0]['file_name']
        pc_im_path = coco.loadImgs(img_id)[0]['pc_file_name']

        pc_im_path = pc_im_path.replace('imagepc', 'imagepc_%02d' % cfg.DATASETS.RADAR_IMAGE_RADIUS)

        img = Image.open(os.path.join(self.root, im_path)).convert('RGB')
        pc_img = Image.open(os.path.join(self.root, pc_im_path)).convert('RGB')

        # if 'R' in cfg.INPUT.PC_MODE:
        #     pc_rcs_img = Image.open(os.path.join(self.root, pc_im_path.replace('.png', '_rcs.png'))).convert('RGB')
        # else:
        #     pc_rsc_img = None

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, pc_img, target = self._transforms(img, pc_img, target)

        return img, pc_img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
