/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// hipDNN-based DNN support for the ROCm backend.
//
// hipDNN is AMD's cuDNN-equivalent library. It exposes both a C "backend"
// descriptor API (mirroring cudnnBackend*) and a C++ "frontend" graph API
// (mirroring cudnn_frontend::graph::Graph). This header provides a thin wrapper
// around the frontend graph API specialized for convolution, playing the same
// role that `CudnnGraph` plays in the CUDA backend
// (xla/stream_executor/cuda/cuda_dnn.cc).
//
// The heavyweight hipdnn_frontend headers are intentionally confined to the .cc
// via the PIMPL idiom so that including this header stays cheap and does not
// leak hipDNN types into the rest of the tree.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_HIPDNN_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_HIPDNN_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace gpu {

// A self-contained wrapper around a hipDNN frontend graph describing a single
// convolution operation (forward, data gradient, or filter gradient).
//
// Lifecycle mirrors the cuDNN frontend flow used by CudnnGraph:
//   1. Create()           - describe tensors + conv op (no device work).
//   2. Prepare(stream)     - compile an execution plan via hipDNN heuristics.
//   3. GetWorkspaceSize()  - query scratch bytes required by the plan.
//   4. Execute(stream,...) - run the plan against device buffers.
//
// Dims/strides use the logical [batch, feature, spatial...] ordering with a
// matching stride vector per tensor, as produced from XLA's
// dnn::BatchDescriptor / dnn::FilterDescriptor. `padding`, `conv_stride` and
// `dilation` carry one entry per spatial dimension.
class HipdnnConvGraph {
 public:
  enum class ConvKind {
    kForward,         // y  = conv(x, w)
    kBackwardData,    // dx = conv_dgrad(dy, w)
    kBackwardFilter,  // dw = conv_wgrad(dy, x)
  };

  HipdnnConvGraph(HipdnnConvGraph&&) noexcept;
  HipdnnConvGraph& operator=(HipdnnConvGraph&&) noexcept;
  ~HipdnnConvGraph();

  static absl::StatusOr<HipdnnConvGraph> Create(
      ConvKind kind, dnn::DataType input_type, dnn::DataType output_type,
      absl::Span<const int64_t> input_dims,
      absl::Span<const int64_t> input_strides,
      absl::Span<const int64_t> filter_dims,
      absl::Span<const int64_t> filter_strides,
      absl::Span<const int64_t> output_dims,
      absl::Span<const int64_t> output_strides,
      absl::Span<const int64_t> padding,
      absl::Span<const int64_t> conv_stride,
      absl::Span<const int64_t> dilation);

  // Compiles an execution plan for the graph. Must be called before
  // GetWorkspaceSize()/Execute(). `stream` supplies the target device/stream.
  absl::Status Prepare(Stream* stream);

  // Scratch memory in bytes required by the compiled plan.
  absl::StatusOr<size_t> GetWorkspaceSize() const;

  // Executes the compiled plan. The three buffers are passed by *role*
  // regardless of ConvKind:
  //   x : input activation  (x for fwd/wgrad, dx output for dgrad)
  //   w : filter            (w for fwd/dgrad, dw output for wgrad)
  //   y : output activation (y output for fwd, dy input for dgrad/wgrad)
  absl::Status Execute(Stream* stream, DeviceAddressBase x, DeviceAddressBase w,
                       DeviceAddressBase y, DeviceAddressBase scratch);

 private:
  struct Impl;
  explicit HipdnnConvGraph(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_HIPDNN_H_
