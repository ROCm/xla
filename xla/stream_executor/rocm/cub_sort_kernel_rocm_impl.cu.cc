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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hipcub/backend/rocprim/device/device_radix_sort.hpp"
#include "rocm/include/rocprim/device/device_segmented_radix_sort.hpp"
#include "rocm/include/rocprim/iterator/counting_iterator.hpp"
#include "rocm/include/rocprim/iterator/transform_iterator.hpp"
#include "rocm/include/rocprim/rocprim_version.hpp"
#include "rocm/include/rocprim/thread/radix_key_codec.hpp"
#include "rocm/include/rocprim/type_traits.hpp"
#include "tsl/platform/bfloat16.h"
#include "xla/stream_executor/rocm/cub_sort_kernel_rocm.h"
#include "xla/stream_executor/rocm/rocm_status.h"

// Required for sorting Eigen::half and bfloat16.
namespace rocprim {

namespace traits {

template <>
struct define<Eigen::half> {
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7C00, 0x03FF>;
  using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
  using number_format = rocprim::traits::number_format::values<
      traits::number_format::kind::floating_point_type>;
};

template <>
struct define<tsl::bfloat16> {
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
  using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
  using number_format = rocprim::traits::number_format::values<
      traits::number_format::kind::floating_point_type>;
};

}  // namespace traits

};  // namespace rocprim

namespace stream_executor {
namespace rocm {
namespace {

// Start offset of segment `i` of a batch of equally sized segments. The
// segment offsets are computed on the fly on the device, so no offsets array
// has to be copied from the host before each sort.
struct SegmentOffset {
  unsigned int segment_size;
  __host__ __device__ unsigned int operator()(unsigned int i) const {
    return i * segment_size;
  }
};

using SegmentOffsetIterator =
    rocprim::transform_iterator<rocprim::counting_iterator<unsigned int>,
                                SegmentOffset, unsigned int>;

SegmentOffsetIterator SegmentBegins(unsigned int segment_size) {
  return SegmentOffsetIterator(rocprim::counting_iterator<unsigned int>(0),
                               SegmentOffset{segment_size});
}

SegmentOffsetIterator SegmentEnds(unsigned int segment_size) {
  return SegmentOffsetIterator(rocprim::counting_iterator<unsigned int>(1),
                               SegmentOffset{segment_size});
}

// rocPRIM's segmented radix sort partitions the segments by size once there
// are at least `partitioning_threshold` of them (a few to a few thousand,
// depending on the architecture): a kernel classifies the segments, and the
// segment counts are copied back to the host with a blocking
// hipMemcpyWithStream so that it can size the launches of the per-class
// kernels. That copy waits for all the work queued before the sort on the
// stream, so the host loses its lead over the GPU at every batched sort. XLA's
// batched sorts have segments of equal size, which all fall in the same class,
// so partitioning gains nothing. This config keeps rocPRIM's tuned parameters
// for the target, but never partitions; the unpartitioned kernel still sorts
// short segments with warp sort.
//
// rocPRIM picks the parameters for a config type with `get_config`, which it
// calls unqualified, so the overload below (found by argument-dependent lookup
// on the config type) takes precedence over the generic one. Older rocPRIM
// versions select configs differently and keep the default config, with the
// host synchronization.
struct UnpartitionedSegmentedRadixSortConfig {};

#if ROCPRIM_VERSION >= 400600
using SegmentedSortConfig = UnpartitionedSegmentedRadixSortConfig;

template <class Selector>
constexpr typename Selector::param_type get_config(
    UnpartitionedSegmentedRadixSortConfig, rocprim::detail::target t) {
  typename Selector::param_type params =
      rocprim::detail::default_select_config<Selector>(t);
  params.warp_sort_config.partitioning_threshold =
      std::numeric_limits<unsigned int>::max();
  return params;
}

// Resolves `get_config` the way rocPRIM does (unqualified, from its detail
// namespace), so the build fails if rocPRIM stops picking the overload above.
constexpr unsigned int SelectedPartitioningThreshold() {
  using rocprim::detail::get_config;
  return get_config<rocprim::detail::segmented_radix_sort_config_selector<
      float, rocprim::empty_type>>(SegmentedSortConfig{},
                                   rocprim::detail::target{})
      .warp_sort_config.partitioning_threshold;
}
static_assert(SelectedPartitioningThreshold() ==
                  std::numeric_limits<unsigned int>::max(),
              "rocPRIM does not use UnpartitionedSegmentedRadixSortConfig");
#else
using SegmentedSortConfig = rocprim::default_config;
#endif

template <typename KeyT>
absl::Status CubSortKeysImpl(void* d_temp_storage, size_t& temp_bytes,
                             const void* d_keys_in, void* d_keys_out,
                             size_t num_items, bool descending,
                             hipStream_t stream) {
  auto err =
      descending
          ? hipcub::DeviceRadixSort::SortKeysDescending<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream)
          : hipcub::DeviceRadixSort::SortKeys<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

template <typename KeyT>
absl::Status CubSortKeysImpl(void* d_temp_storage, size_t& temp_bytes,
                             const void* d_keys_in, void* d_keys_out,
                             size_t num_items, bool descending,
                             size_t batch_size, hipStream_t stream) {
  if (batch_size == 1) {
    return CubSortKeysImpl<KeyT>(d_temp_storage, temp_bytes, d_keys_in,
                                 d_keys_out, num_items, descending, stream);
  }
  const unsigned int segment_size = num_items / batch_size;
  auto err =
      descending
          ? rocprim::segmented_radix_sort_keys_desc<SegmentedSortConfig>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, batch_size,
                SegmentBegins(segment_size), SegmentEnds(segment_size),
                /*begin_bit=*/0, /*end_bit=*/sizeof(KeyT) * 8, stream)
          : rocprim::segmented_radix_sort_keys<SegmentedSortConfig>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, batch_size,
                SegmentBegins(segment_size), SegmentEnds(segment_size),
                /*begin_bit=*/0, /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairsImpl(void* d_temp_storage, size_t& temp_bytes,
                              const void* d_keys_in, void* d_keys_out,
                              const void* d_values_in, void* d_values_out,
                              size_t num_items, bool descending,
                              hipStream_t stream) {
  auto err =
      descending
          ? hipcub::DeviceRadixSort::SortPairsDescending<KeyT, ValT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream)
          : hipcub::DeviceRadixSort::SortPairs<KeyT, ValT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairsImpl(void* d_temp_storage, size_t& temp_bytes,
                              const void* d_keys_in, void* d_keys_out,
                              const void* d_values_in, void* d_values_out,
                              size_t num_items, bool descending,
                              size_t batch_size, hipStream_t stream) {
  if (batch_size == 1) {
    return CubSortPairsImpl<KeyT, ValT>(d_temp_storage, temp_bytes, d_keys_in,
                                        d_keys_out, d_values_in, d_values_out,
                                        num_items, descending, stream);
  }
  const unsigned int segment_size = num_items / batch_size;
  auto err =
      descending
          ? rocprim::segmented_radix_sort_pairs_desc<SegmentedSortConfig>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, batch_size,
                SegmentBegins(segment_size), SegmentEnds(segment_size),
                /*begin_bit=*/0, /*end_bit=*/sizeof(KeyT) * 8, stream)
          : rocprim::segmented_radix_sort_pairs<SegmentedSortConfig>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, batch_size,
                SegmentBegins(segment_size), SegmentEnds(segment_size),
                /*begin_bit=*/0, /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

}  // namespace

// Template instantiations for CubSortKeys
template <typename KeyT>
absl::Status CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending, size_t batch_size,
                         hipStream_t stream) {
  return CubSortKeysImpl<KeyT>(d_temp_storage, temp_bytes, d_keys_in,
                               d_keys_out, num_items, descending, batch_size,
                               stream);
}

// Template instantiations for CubSortPairs
template <typename KeyT, typename ValT>
absl::Status CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                          const void* d_keys_in, void* d_keys_out,
                          const void* d_values_in, void* d_values_out,
                          size_t num_items, bool descending, size_t batch_size,
                          hipStream_t stream) {
  return CubSortPairsImpl<KeyT, ValT>(
      d_temp_storage, temp_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, descending, batch_size, stream);
}

#define XLA_CUB_DEFINE_SORT_KEYS(type)                                        \
  template absl::Status CubSortKeys<type>(void*, size_t&, const void*, void*, \
                                          size_t, bool, size_t, hipStream_t);

#define XLA_CUB_DEFINE_SORT_PAIRS(type1, type2)                             \
  template absl::Status CubSortPairs<type1, type2>(                         \
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool, \
      size_t, hipStream_t);

// Floating point types.
#ifdef CUB_TYPE_BF16
XLA_CUB_DEFINE_SORT_KEYS(hip_bfloat16)
#endif
#ifdef CUB_TYPE_F16
XLA_CUB_DEFINE_SORT_KEYS(__half)
#endif
#ifdef CUB_TYPE_F32
XLA_CUB_DEFINE_SORT_KEYS(float)
#endif
#ifdef CUB_TYPE_F64
XLA_CUB_DEFINE_SORT_KEYS(double)
#endif

// Signed integer types.
#ifdef CUB_TYPE_S8
XLA_CUB_DEFINE_SORT_KEYS(int8_t)
#endif
#ifdef CUB_TYPE_S16
XLA_CUB_DEFINE_SORT_KEYS(int16_t)
#endif
#ifdef CUB_TYPE_S32
XLA_CUB_DEFINE_SORT_KEYS(int32_t)
#endif
#ifdef CUB_TYPE_S64
XLA_CUB_DEFINE_SORT_KEYS(int64_t)
#endif

// Unsigned integer types.
#ifdef CUB_TYPE_U8
XLA_CUB_DEFINE_SORT_KEYS(uint8_t)
#endif
#ifdef CUB_TYPE_U16
XLA_CUB_DEFINE_SORT_KEYS(uint16_t)
#endif
#ifdef CUB_TYPE_U32
XLA_CUB_DEFINE_SORT_KEYS(uint32_t)
#endif
#ifdef CUB_TYPE_U64
XLA_CUB_DEFINE_SORT_KEYS(uint64_t)
#endif

// Pairs with 8-bit key.
#ifdef CUB_TYPE_U8_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint8_t, uint16_t)
#endif
#ifdef CUB_TYPE_U8_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint8_t, uint32_t)
#endif
#ifdef CUB_TYPE_U8_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint8_t, uint64_t)
#endif

// Pairs with 16-bit key.
#ifdef CUB_TYPE_U16_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint16_t, uint16_t)
#endif
#ifdef CUB_TYPE_U16_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint16_t, uint32_t)
#endif
#ifdef CUB_TYPE_U16_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint16_t, uint64_t)
#endif

// Pairs with signed 32-bit key.
#ifdef CUB_TYPE_S32_B16
XLA_CUB_DEFINE_SORT_PAIRS(int32_t, uint16_t)
#endif
#ifdef CUB_TYPE_S32_B32
XLA_CUB_DEFINE_SORT_PAIRS(int32_t, uint32_t)
#endif
#ifdef CUB_TYPE_S32_B64
XLA_CUB_DEFINE_SORT_PAIRS(int32_t, uint64_t)
#endif

// Pairs with unsigned 32-bit key.
#ifdef CUB_TYPE_U32_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint32_t, uint16_t)
#endif
#ifdef CUB_TYPE_U32_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint32_t, uint32_t)
#endif
#ifdef CUB_TYPE_U32_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint32_t, uint64_t)
#endif

// Pairs with 64-bit key.
#ifdef CUB_TYPE_U64_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint64_t, uint16_t)
#endif
#ifdef CUB_TYPE_U64_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint64_t, uint32_t)
#endif
#ifdef CUB_TYPE_U64_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint64_t, uint64_t)
#endif

// Pairs with f32 key.
#ifdef CUB_TYPE_F32_B16
XLA_CUB_DEFINE_SORT_PAIRS(float, uint16_t)
#endif
#ifdef CUB_TYPE_F32_B32
XLA_CUB_DEFINE_SORT_PAIRS(float, uint32_t)
#endif
#ifdef CUB_TYPE_F32_B64
XLA_CUB_DEFINE_SORT_PAIRS(float, uint64_t)
#endif

}  // namespace rocm
}  // namespace stream_executor
