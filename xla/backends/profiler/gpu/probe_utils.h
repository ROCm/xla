#ifndef XLA_BACKENDS_PROFILER_GPU_PROBE_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_PROBE_UTILS_H_

#include <cstdint>
#include <vector>
#include "absl/container/flat_hash_map.h"

namespace xla::profiler {

// Forward declaration - defined in svm_wrapper.h
struct Point;

}  // namespace xla::profiler

namespace xla::profiler::probe_info {

struct ProbePair {
  uint64_t pt1_tx, pt2_tx, pt1_rx, pt2_rx;
  uint64_t pr1_tx, pr2_tx, pr1_rx, pr2_rx;
  bool pure_pt, pure_pr;
  ProbePair() : pt1_tx(0), pt2_tx(0), pt1_rx(0), pt2_rx(0), pr1_tx(0), pr2_tx(0), pr1_rx(0), pr2_rx(0), pure_pt(false), pure_pr(false) {}
};

}  // namespace xla::profiler::probe_info

namespace xla::profiler::probe_utils {

std::vector<Point> convert_probe_pairs_to_xy_pairs(
    const absl::flat_hash_map<uint32_t, probe_info::ProbePair>& probe_pairs, 
    double threshold = 300000, 
    bool adaptive = false
);

}  // namespace xla::profiler::probe_utils

#endif  // XLA_BACKENDS_PROFILER_GPU_PROBE_UTILS_H_
