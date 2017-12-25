// ------------------------------------------------------------------
// Deformable Convolutional Networks
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
// ------------------------------------------------------------------

#include "gpu_nms.hpp"
#include <vector>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}


__global__ void ious_kernel(const int n_boxes, const float * dev_boxes, float * ious) {
  CUDA_1D_KERNEL_LOOP(out_idx, n_boxes * n_boxes) {
    const int idx_y = out_idx / n_boxes;
    const int idx_x = out_idx % n_boxes;

    float iou = 1;
    if (idx_y != idx_x) {
      iou = devIoU(dev_boxes + idx_y * 5, dev_boxes + idx_x * 5);
    }
    ious[idx_y * n_boxes + idx_x] = iou;
  }
}


__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      float iou = devIoU(cur_box, block_boxes + i * 5);
      // ious[cur_box_idx * n_boxes + (threadsPerBlock * col_start) + i] = iou;
      if (iou > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}


void _soft_nms(int* keep_out, int* num_out, float* boxes_host, const int boxes_num,
          const int boxes_dim, float sigma, float nms_overlap_thresh, float soft_threshold, int method, int device_id) {

  _set_device(device_id);
  
  float* boxes_dev = NULL;
  float* iou_dev = NULL;

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&iou_dev,
                        boxes_num * boxes_num * sizeof(float)));
  
  const int total_count = boxes_num * boxes_num;
  const int thread_per_block = 512;
  const int block_count = (total_count + thread_per_block - 1) / thread_per_block;

  ious_kernel<<<block_count, thread_per_block>>>(boxes_num, boxes_dev, iou_dev);

  // soft nms
  float iou_host[boxes_num * boxes_num];
  CUDA_CHECK(cudaMemcpy(iou_host,
                        iou_dev,
                        sizeof(float) * boxes_num * boxes_num,
                        cudaMemcpyDeviceToHost));
  
  int boxes_ind[boxes_num];
  // float scores[boxes_num];
  for (int i = 0; i < boxes_num; i++) {
    boxes_ind[i] = i;
    // scores[i] = boxes_host[i * 5 + 4];
  }
  int N = boxes_num;  // remaining boxes
  for (int i = 0; i < N; i++) {

    // get max box
    float maxscore = boxes_host[boxes_ind[i] * 5 + 4];
    int maxpos = i;
    int pos = i + 1;
    while (pos < N) {
      float tmp_score = boxes_host[boxes_ind[pos] * 5 + 4];
      if (maxscore < tmp_score) {
        maxscore = tmp_score;
        maxpos = pos;
      }
      pos++;
    }

    // swap ith box with position of max box
    int tmp_ind = boxes_ind[i];
    boxes_ind[i] = boxes_ind[maxpos];
    boxes_ind[maxpos] = tmp_ind;
    
    // NMS iterations, note that N changes if detection boxes fall below threshold
    int max_ind = boxes_ind[i];
    pos = i + 1;
    while (pos < N) {
      int curr_ind = boxes_ind[pos];
      float iou = iou_host[curr_ind * boxes_num + max_ind];
      // if (abs(iou_host[curr_ind * boxes_num + max_ind] - iou_host[max_ind * boxes_num + curr_ind]) > 1e-4) {
      //   std::cout << iou_host[curr_ind * boxes_num + max_ind] << ", " << iou_host[max_ind * boxes_num + curr_ind] << std::endl;
      //   // iou = iou_host[curr_ind * boxes_num + max_ind] + iou_host[max_ind * boxes_num + curr_ind];
      // }
      
      if (iou > 0) {
        float weight = 0;
        switch(method) {
          case 1: // linear
            if (iou > nms_overlap_thresh) {
              weight = 1 - iou;
            } else {
              weight = 1;
            }
            break;
          case 2: // gaussian
            weight = std::exp(-iou * iou / sigma);
            break;
          default:
            if (iou > nms_overlap_thresh) {
              weight = 0;
            }
            else {
              weight = 1;
            }
            break;
        }
        boxes_host[curr_ind * 5 + 4] *= weight;
        if (boxes_host[curr_ind * 5 + 4] < soft_threshold) {
          boxes_ind[pos] = boxes_ind[N-1];
          N--;
          pos--;
        }
      }
      pos++;
    }
  }

  for (int i = 0; i < N; i++) {
    keep_out[i] = boxes_ind[i];
  }
  *num_out = N;
 
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(iou_dev));
}
