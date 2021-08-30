# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import os
from itertools import islice
import numpy as np


def extract_loss_item(loss_str):
    loss_str = loss_str.split('(')
    loss1 = float(loss_str[0].rstrip(' '))
    loss2 = float(loss_str[1].split(')')[0])

    res = [loss1, loss2]

    return res


def extract_loss(log_file_path):
    loss_strs = [line for line in open(log_file_path) if 'fcos_core.trainer INFO: eta:' in line]

    loss_info = []
    for loss_str in loss_strs:
        loss_str = loss_str.split(':')
        info = list()
        info.append(float(loss_str[7].split(' ')[1]))
        info.extend(extract_loss_item(loss_str[8]))
        info.extend(extract_loss_item(loss_str[9]))
        info.extend(extract_loss_item(loss_str[10]))
        info.extend(extract_loss_item(loss_str[11]))
        loss_info.append(info)

    loss_info = sorted(loss_info, key=lambda i: i[0])
    loss_info = np.asarray(loss_info)
    return loss_info.astype(np.float)


def extract_acc(eval_file_path):
    acc_strs = [line for line in open(eval_file_path) if 'Average' in line]

    if len(acc_strs) % 12 == 0:
        all_acc = [float(loss_str.split('=')[-1].rstrip('\n')) for loss_str in acc_strs]
        print(int(len(all_acc) / 12))
        length_to_split = [12, ] * int(len(all_acc) / 12)
        all_acc = iter(all_acc)
        all_acc = [list(islice(all_acc, elem)) for elem in length_to_split]
        all_acc = np.asarray(all_acc)
        return all_acc.astype(np.float)
    else:
        os.sys.exit("Wrong eval file: %s, Exit!" % eval_file_path)


def main():
    dir_root = '/home/citybuster/Projects/FCOS/training_dir'
    sub_dirs = [os.path.join(dir_root, dI) for dI in os.listdir(dir_root) if os.path.isdir(os.path.join(dir_root, dI))]

    # The Loss and Acc format
    # Loss: loss: 1.0316 (1.2633) loss_centerness: 0.6217 (0.6416) loss_cls: 0.2034 (0.3138) loss_reg: 0.2068 (0.3079)
    # Acc: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
    #      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.724
    #      Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.418
    #      Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235s
    #      Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.417
    #      Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.539
    #      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.106
    #      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.427
    #      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
    #      Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350
    #      Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
    #      Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686

    for sub_dir in sub_dirs:
        log_path = os.path.join(sub_dir, 'log.txt')
        eval_path = os.path.join(sub_dir, 'eval.txt')
        if os.path.isfile(log_path) and os.path.isfile(eval_path):
            loss = extract_loss(log_path)
            acc = extract_acc(eval_path)
            np.savetxt(os.path.join(sub_dir, 'extract_log.txt'), loss, fmt='%.4f')
            np.savetxt(os.path.join(sub_dir, 'extract_eval.txt'), acc, fmt='%.4f')


if __name__ == "__main__":
    main()
