import json
import os
import sys
from collections import defaultdict
from concurrent import futures

import cv2
import matplotlib
import numpy as np

# Use a non-interactive backend
matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Print iterations progress (thanks StackOverflow)
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(bar_length * iteration / float(total)))
    bar = '' * filledLength + '-' * (bar_length - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def vis_one_image(im, im_name, output_dir, boxes, thresh=0.508, dpi=200, box_alpha=0.8, show_class=True, ext='png',
                  out_when_no_box=False):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh) and not out_when_no_box:
    #     return

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    if boxes is None:
        sorted_inds = []  # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = boxes[:, 2] * boxes[:, 3]
        sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2, 'obstacle' + ' {:0.2f}'.format(score).lstrip('0'),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')


def run(dic):
    img = cv2.imread(os.path.join('/home/citybuster/Data/nuScenes', dic["file_name"]), cv2.IMREAD_COLOR)[:, :, ::-1]
    basename = os.path.splitext(os.path.basename(dic["file_name"]))[0]
    image_id = dic['id']
    cls_box_i = []
    for res in pred_by_image[image_id]:
        item = list()
        score = res['score']
        box = res['bbox']
        item.extend(box)
        item.append(score)
        cls_box_i.append(item)
    cls_box_i = np.asarray(cls_box_i)
    vis_one_image(img, basename, '/home/citybuster/Projects/FCOS/' + config_name, cls_box_i)


if __name__ == "__main__":

    config_name = 'fcos_imprv_R_50_FPN_1x_IMG'
    json_file_path = '/home/citybuster/Projects/FCOS/training_dir/fcos_imprv_R_50_FPN_1x_IMG/inference-model_0090000.pth/nuscenes_test_cocostyle/bbox.json'
    # json_file_path = '/home/citybuster/Projects/FCOS/training_dir/fcos_imprv_R_50_FPN_1x_ATTMIX_135_Circle_07' \
    #                  '/inference-model_0037500.pth/nuscenes_test_cocostyle/bbox.json'
    with open(json_file_path, "r") as f:
        predictions = json.load(f)
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    with open('/home/citybuster/Data/nuScenes/v1.0-trainval/gt_fcos_coco_test_v2.json', 'r') as f:
        dicts = json.load(f)
        dicts = dicts['images']

    print("Generating 2D detection results")
    num_threads = 40
    num_tokens = len(dicts)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(run, dic) for dic in
              dicts]
        for i, f in enumerate(futures.as_completed(fs)):
            # Write progress to error so that it can be seen
            print_progress(i, num_tokens, prefix='VisDetection', suffix='Done ', bar_length=40)
