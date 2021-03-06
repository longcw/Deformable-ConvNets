# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

cdef extern from "gpu_nms.hpp":
    void _soft_nms(np.int32_t*, int*, np.float32_t*, int, int,
                   float, float, float, int, int);

def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 4]
    cdef np.ndarray[np.int32_t, ndim=1] \
        order = scores.argsort()[::-1].astype(np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :]
    _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return list(order[keep])


def gpu_soft_nms(np.ndarray[np.float32_t, ndim=2] dets,
                 float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0, np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)

    _soft_nms(&keep[0], &num_out, &dets[0, 0], boxes_num, boxes_dim, sigma, Nt, threshold, method, device_id)
    keep = keep[:num_out]
    return keep