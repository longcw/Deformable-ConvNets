# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/rfcn/cfgs/rfcn_coco_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes, show_boxes_cv2
from utils.tictoc import tic, toc
from utils.timer import Timer
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper, soft_nms_wrapper, gpu_soft_nms_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False,
                        action='store_true')

    args = parser.parse_args()
    return args


args = parse_args()


def run_detection(im_root, result_root, conf_threshold):
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    print('detection in {}'.format(im_root))
    im_names = sorted(os.listdir(im_root))

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = []
    for idx, im_name in enumerate(im_names[:2]):
        im_file = os.path.join(im_root, im_name)
        im = cv2.imread(im_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        data.append({'data': im_tensor, 'im_info': im_info})

    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    arg_params, aux_params = load_param(
        cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'),
        config.TEST.test_epoch, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    # nms = gpu_nms_wrapper(config.TEST.NMS, 0)
    # nms = soft_nms_wrapper(config.TEST.NMS, method=2)
    nms = gpu_soft_nms_wrapper(config.TEST.NMS, method=2, device_id=0)

    nms_t = Timer()
    for idx, im_name in enumerate(im_names):
        im_file = os.path.join(im_root, im_name)
        im = cv2.imread(im_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        origin_im = im.copy()

        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        # input
        data = [mx.nd.array(im_tensor), mx.nd.array(im_info)]
        data_batch = mx.io.DataBatch(data=[data], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            nms_t.tic()
            keep = nms(cls_dets)
            nms_t.toc()
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.2f}ms'.format(im_name, toc() * 1000)
        print 'nms: {:.2f}ms'.format(nms_t.total_time * 1000)
        nms_t.clear()

        # save results
        person_dets = dets_nms[0]
        with open(os.path.join(result_root, '{:04d}.txt'.format(idx)), 'w') as f:
            f.write('{}\n'.format(len(person_dets)))
            for det in person_dets:
                x1, y1, x2, y2, s = det
                w = x2 - x1
                h = y2 - y1
                f.write('0 {} {} {} {} {}\n'.format(s, w, h, x1, y1))

        # visualize
        im = origin_im
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = show_boxes_cv2(im, dets_nms, classes, 1)

        cv2.imshow('det', im)
        cv2.waitKey(1)


if __name__ == '__main__':

    def mkdirs(path):
        if not os.path.exists(path):
            os.makedirs(path)

    conf_threshold = 0.7
    im_root = '/data/KITTI/training/image_02'
    result_root = '/data/KITTI/training/rfcn_dcn_dets_02'
    for seq_name in os.listdir(im_root):
        dst_root = os.path.join(result_root, seq_name)
        mkdirs(dst_root)
        run_detection(os.path.join(im_root, seq_name), dst_root, conf_threshold)
