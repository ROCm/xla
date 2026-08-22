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

#include "xla/stream_executor/rocm/rocm_hipdnn.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"

// glibc's <sys/sysmacros.h> (pulled in transitively by the HIP runtime headers,
// especially under hipcc) defines `major` and `minor` as function-like macros.
// These collide with identifiers of the same name used inside the hipDNN
// frontend headers (e.g. VersionUtils.hpp), so undefine them first.
#ifdef major
#undef major
#endif
#ifdef minor
#undef minor
#endif

// hipDNN frontend (C++ graph API). Confined to this translation unit. JSON and
// SDPA are disabled to avoid pulling extra third-party dependencies into the
// build; convolution does not need them.
#include "hipdnn_frontend.hpp"

namespace stream_executor {
namespace gpu {
namespace {

namespace fe = hipdnn_frontend;

// Stable tensor UIDs for the variant pack. Roles are consistent across conv
// kinds; only which of x/w is the graph output changes.
constexpr int64_t kXUid = 1;  // input activation (x) or its gradient (dx)
constexpr int64_t kWUid = 2;  // filter (w) or its gradient (dw)
constexpr int64_t kYUid = 3;  // output activation (y) or its gradient (dy)

absl::Status ToStatus(const fe::Error& err, absl::string_view what) {
  if (err.code == fe::error_code_t::OK) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(what, ": hipDNN frontend error [",
                                          fe::to_string(err.code), "] ",
                                          err.err_msg));
}

absl::StatusOr<fe::DataType> ToHipdnnDataType(dnn::DataType dtype) {
  switch (dtype) {
    case dnn::DataType::kFloat:
      return fe::DataType::FLOAT;
    case dnn::DataType::kDouble:
      return fe::DataType::DOUBLE;
    case dnn::DataType::kHalf:
      return fe::DataType::HALF;
    case dnn::DataType::kBF16:
      return fe::DataType::BFLOAT16;
    case dnn::DataType::kInt8:
      return fe::DataType::INT8;
    case dnn::DataType::kInt32:
      return fe::DataType::INT32;
    case dnn::DataType::kF8E4M3FN:
      return fe::DataType::FP8_E4M3;
    case dnn::DataType::kF8E5M2:
      return fe::DataType::FP8_E5M2;
    case dnn::DataType::kF8E4M3FNUZ:
      return fe::DataType::FP8_E4M3_FNUZ;
    case dnn::DataType::kF8E5M2FNUZ:
      return fe::DataType::FP8_E5M2_FNUZ;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "hipDNN: unsupported dnn::DataType ", static_cast<int>(dtype)));
  }
}

hipStream_t HipStreamHandle(Stream* stream) {
  return static_cast<hipStream_t>(stream->platform_specific_handle().stream);
}

std::vector<int64_t> ToVec(absl::Span<const int64_t> s) {
  return std::vector<int64_t>(s.begin(), s.end());
}

}  // namespace

struct HipdnnConvGraph::Impl {
  fe::graph::Graph graph;
  fe::HipdnnHandlePtr handle;  // owns the hipDNN handle used to build the plan.
  ConvKind kind = ConvKind::kForward;
};

HipdnnConvGraph::HipdnnConvGraph(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
HipdnnConvGraph::HipdnnConvGraph(HipdnnConvGraph&&) noexcept = default;
HipdnnConvGraph& HipdnnConvGraph::operator=(HipdnnConvGraph&&) noexcept =
    default;
HipdnnConvGraph::~HipdnnConvGraph() = default;

absl::StatusOr<HipdnnConvGraph> HipdnnConvGraph::Create(
    ConvKind kind, dnn::DataType input_type, dnn::DataType output_type,
    absl::Span<const int64_t> input_dims,
    absl::Span<const int64_t> input_strides,
    absl::Span<const int64_t> filter_dims,
    absl::Span<const int64_t> filter_strides,
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> output_strides,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> conv_stride,
    absl::Span<const int64_t> dilation) {
  auto impl = std::make_unique<Impl>();
  impl->kind = kind;

  ABSL_ASSIGN_OR_RETURN(fe::DataType in_dt, ToHipdnnDataType(input_type));
  ABSL_ASSIGN_OR_RETURN(fe::DataType out_dt, ToHipdnnDataType(output_type));

  fe::graph::Graph& g = impl->graph;
  g.set_io_data_type(in_dt).set_compute_data_type(fe::DataType::FLOAT);

  const std::vector<int64_t> pad = ToVec(padding);
  const std::vector<int64_t> str = ToVec(conv_stride);
  const std::vector<int64_t> dil = ToVec(dilation);

  switch (kind) {
    case ConvKind::kForward: {
      auto x = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                            .set_dim(ToVec(input_dims))
                                            .set_stride(ToVec(input_strides))
                                            .set_data_type(in_dt)
                                            .set_uid(kXUid)
                                            .set_name("x"));
      auto w = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                            .set_dim(ToVec(filter_dims))
                                            .set_stride(ToVec(filter_strides))
                                            .set_data_type(in_dt)
                                            .set_uid(kWUid)
                                            .set_name("w"));
      auto y = g.conv_fprop(x, w,
                            fe::graph::ConvFpropAttributes()
                                .set_padding(pad)
                                .set_stride(str)
                                .set_dilation(dil));
      y->set_dim(ToVec(output_dims))
          .set_stride(ToVec(output_strides))
          .set_data_type(out_dt)
          .set_output(true)
          .set_uid(kYUid)
          .set_name("y");
      break;
    }
    case ConvKind::kBackwardData: {
      auto dy = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                             .set_dim(ToVec(output_dims))
                                             .set_stride(ToVec(output_strides))
                                             .set_data_type(in_dt)
                                             .set_uid(kYUid)
                                             .set_name("dy"));
      auto w = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                            .set_dim(ToVec(filter_dims))
                                            .set_stride(ToVec(filter_strides))
                                            .set_data_type(in_dt)
                                            .set_uid(kWUid)
                                            .set_name("w"));
      auto dx = g.conv_dgrad(dy, w,
                             fe::graph::ConvDgradAttributes()
                                 .set_padding(pad)
                                 .set_stride(str)
                                 .set_dilation(dil));
      dx->set_dim(ToVec(input_dims))
          .set_stride(ToVec(input_strides))
          .set_data_type(out_dt)
          .set_output(true)
          .set_uid(kXUid)
          .set_name("dx");
      break;
    }
    case ConvKind::kBackwardFilter: {
      auto dy = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                             .set_dim(ToVec(output_dims))
                                             .set_stride(ToVec(output_strides))
                                             .set_data_type(in_dt)
                                             .set_uid(kYUid)
                                             .set_name("dy"));
      auto x = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                            .set_dim(ToVec(input_dims))
                                            .set_stride(ToVec(input_strides))
                                            .set_data_type(in_dt)
                                            .set_uid(kXUid)
                                            .set_name("x"));
      auto dw = g.conv_wgrad(dy, x,
                             fe::graph::ConvWgradAttributes()
                                 .set_padding(pad)
                                 .set_stride(str)
                                 .set_dilation(dil));
      dw->set_dim(ToVec(filter_dims))
          .set_stride(ToVec(filter_strides))
          .set_data_type(out_dt)
          .set_output(true)
          .set_uid(kWUid)
          .set_name("dw");
      break;
    }
  }

  return HipdnnConvGraph(std::move(impl));
}

absl::Status HipdnnConvGraph::Prepare(Stream* stream) {
  fe::HipdnnHandlePtr handle;
  ABSL_RETURN_IF_ERROR(ToStatus(
      fe::createHipdnnHandle(handle, HipStreamHandle(stream)),
      "Failed to create hipDNN handle"));

  fe::graph::Graph& g = impl_->graph;
  ABSL_RETURN_IF_ERROR(ToStatus(g.validate(), "hipDNN graph validation failed"));
  ABSL_RETURN_IF_ERROR(
      ToStatus(g.build(*handle), "hipDNN graph build failed"));

  impl_->handle = std::move(handle);
  return absl::OkStatus();
}

absl::StatusOr<size_t> HipdnnConvGraph::GetWorkspaceSize() const {
  int64_t workspace_size = 0;
  ABSL_RETURN_IF_ERROR(ToStatus(impl_->graph.get_workspace_size(workspace_size),
                              "hipDNN get_workspace_size failed"));
  return static_cast<size_t>(workspace_size);
}

absl::Status HipdnnConvGraph::Execute(Stream* stream, DeviceAddressBase x,
                                      DeviceAddressBase w, DeviceAddressBase y,
                                      DeviceAddressBase scratch) {
  // Execute on the SAME handle used by Prepare() to build the plan. hipDNN's
  // MIOpen engine plugin registers its convolution invoker (via MIOpen Find)
  // on the handle at build time; executing on a freshly created handle fails
  // with "No invoker was registered ... Was find executed?" for solvers that
  // rely on that registration. Rebind the build handle to the (possibly
  // different) execution stream instead of creating a new one.
  if (impl_->handle == nullptr) {
    return absl::FailedPreconditionError(
        "HipdnnConvGraph::Execute called before Prepare()");
  }
  ABSL_RETURN_IF_ERROR(ToStatus(
      fe::setHipdnnHandleStream(impl_->handle, HipStreamHandle(stream)),
      "Failed to bind hipDNN handle to execution stream"));

  std::unordered_map<int64_t, void*> variant_pack = {
      {kXUid, x.opaque()},
      {kWUid, w.opaque()},
      {kYUid, y.opaque()},
  };

  ABSL_RETURN_IF_ERROR(ToStatus(
      impl_->graph.execute(*impl_->handle, variant_pack, scratch.opaque()),
      "hipDNN graph execution failed"));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace stream_executor
