void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);

void _soft_nms(int* keep_out, int* num_out, float* boxes_host, const int boxes_num,
          const int boxes_dim, float sigma, float nms_overlap_thresh, float soft_threshold, int method, int device_id);