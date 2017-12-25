# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import random as rand


def show_boxes(im, dets, classes, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    return im


def show_boxes_cv2(im, dets, classes, scale=1.0):
    def _to_color(indx, base):
        """ return (b, r, g) tuple"""
        base2 = base * base
        b = 2 - indx / base2
        r = 2 - (indx % base2) / base
        g = 2 - (indx % base2) % base
        return b * 127, r * 127, g * 127

    def get_color(indx, cls_num=1):
        if indx >= cls_num:
            return 23 * indx % 255, 47 * indx % 255, 137 * indx % 255
        base = int(np.ceil(pow(cls_num, 1. / 3)))
        return _to_color(indx, base)

    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = tuple([int(round(x)) for x in det[:4] * scale])
            color = get_color(cls_idx, len(classes))

            cv2.rectangle(im, bbox[0:2], bbox[2:4], color, thickness=2)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                cv2.putText(im, '{:s} {:.3f}'.format(cls_name, score), (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)

    return im
