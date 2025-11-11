#include "xla/backends/profiler/gpu/probe_utils.h"
#include "xla/backends/profiler/gpu/svm_wrapper.h"

#include <cstdio>
#include <algorithm>
#include "absl/container/flat_hash_map.h"
#include <cmath>

namespace xla::profiler::probe_utils {

std::vector<Point> convert_probe_pairs_to_xy_pairs(
    const absl::flat_hash_map<uint32_t, probe_info::ProbePair>& probe_pairs, int threshold) {
  std::vector<Point> xy_pairs;
  uint64_t min_pt1_tx = -1;
  for (const auto& [seq_id, probe_pair] : probe_pairs) {
    if(probe_pair.pt1_tx == 0 || probe_pair.pt2_tx == 0 || probe_pair.pt1_rx == 0 || probe_pair.pt2_rx == 0) {
      continue;
    }
    if(probe_pair.pr1_tx == 0 || probe_pair.pr2_tx == 0 || probe_pair.pr1_rx == 0 || probe_pair.pr2_rx == 0) {
      continue;
    }
    if(min_pt1_tx == -1 || probe_pair.pt1_tx < min_pt1_tx) {
      min_pt1_tx = probe_pair.pt1_tx;
    }
  }
  for (const auto& [seq_id, probe_pair] : probe_pairs) {
    if(probe_pair.pt1_tx == 0 || probe_pair.pt2_tx == 0 || probe_pair.pt1_rx == 0 || probe_pair.pt2_rx == 0) {
      continue;
    }
    if(probe_pair.pr1_tx == 0 || probe_pair.pr2_tx == 0 || probe_pair.pr1_rx == 0 || probe_pair.pr2_rx == 0) {
      continue;
    }
    auto type_safe_sub = [](uint64_t a, uint64_t b) -> long long {
      return static_cast<long long>(a) - static_cast<long long>(b);
    };
    if (std::abs(static_cast<long long>(type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx) - 
                                        type_safe_sub(probe_pair.pt2_rx, probe_pair.pt1_rx))) < threshold) {
      xy_pairs.push_back({static_cast<double>(probe_pair.pt1_tx - min_pt1_tx), 
                          static_cast<double>(probe_pair.pt1_rx - probe_pair.pt1_tx), 1});
      xy_pairs.push_back({static_cast<double>(probe_pair.pt2_tx - min_pt1_tx), 
                          static_cast<double>(probe_pair.pt2_rx - probe_pair.pt2_tx), 1});
    }
    if (std::abs(static_cast<long long>(type_safe_sub(probe_pair.pr2_tx, probe_pair.pr1_tx) - 
                                        type_safe_sub(probe_pair.pr2_rx, probe_pair.pr1_rx))) < threshold) {
      xy_pairs.push_back({static_cast<double>(probe_pair.pr1_rx - min_pt1_tx), 
                          -static_cast<double>(probe_pair.pr1_rx - probe_pair.pr1_tx), -1});
      xy_pairs.push_back({static_cast<double>(probe_pair.pr2_rx - min_pt1_tx), 
                          -static_cast<double>(probe_pair.pr2_rx - probe_pair.pr2_tx), -1});
    }
  }
  for (const auto& point : xy_pairs) {
    fprintf(stderr, "x: %f, y: %f, label: %d\n", point.x, point.y, point.label);
  }
  fprintf(stderr, "xy_pairs size: %zu\n", xy_pairs.size());
  return xy_pairs;
}

}  // namespace xla::profiler::probe_utils
