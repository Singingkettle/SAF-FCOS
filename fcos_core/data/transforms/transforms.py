# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, pc_image, target):
        for t in self.transforms:
            image, pc_image, target = t(image, pc_image, target)
        return image, pc_image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, pc_image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        pc_image = F.resize(pc_image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(image.size)
        return image, pc_image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, pc_image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            pc_image = F.hflip(pc_image)
            target = target.transpose(0)
        return image, pc_image, target


class ToTensor(object):
    def __call__(self, image, pc_image, target):
        return F.to_tensor(image), F.to_tensor(pc_image), target


class Normalize(object):
    def __init__(self, mean, std, pc_mean, pc_std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.pc_mean = pc_mean
        self.pc_std = pc_std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, pc_image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
            pc_image = pc_image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        pc_image = F.normalize(pc_image, mean=self.pc_mean, std=self.pc_std)
        if target is None:
            return image, pc_image
        return image, pc_image, target