// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <ATen/ATen.h>
#include <torch/extension.h>

at::Tensor ROIAlignRotated_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

// Interface for Python
inline at::Tensor ROIAlignRotated_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio) {
  if (input.is_cuda()) {
    return ROIAlignRotated_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);
  }
  return ROIAlignRotated_forward_cpu(
      input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

inline at::Tensor ROIAlignRotated_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t sampling_ratio) {
  if (grad.is_cuda()) {
    return ROIAlignRotated_backward_cuda(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio);
  }
  return ROIAlignRotated_backward_cpu(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ROIAlignRotated_forward, "roi_align_rotated_forward");
  m.def("backward", &ROIAlignRotated_backward, "roi_align_rotated_backward");
}
