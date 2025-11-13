#include "xla/backends/profiler/gpu/probe_utils.h"
#include "xla/backends/profiler/gpu/svm_wrapper.h"

#include <cstdio>
#include <algorithm>
#include "absl/container/flat_hash_map.h"
#include <cmath>
#include "absl/log/log.h"

namespace xla::profiler::probe_utils {

std::vector<Point> convert_probe_pairs_to_xy_pairs(
    const absl::flat_hash_map<uint32_t, probe_info::ProbePair>& probe_pairs, 
    double threshold,
    bool adaptive
) {
  std::vector<Point> xy_pairs;
  uint64_t min_pt1_tx = -1;
  auto _threshold = threshold;
  auto good_pos_pairs = 0, good_neg_pairs = 0;
  auto filtered_pos_pairs = 0;
  auto filtered_neg_pairs = 0;
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
    auto signed_sub = [](uint64_t a, uint64_t b) -> long long {
      auto sign = 1;
      if(a > b){
        sign = -1;
        std::swap(a, b);
      }
      return sign * (b - a);
    };
    auto abs_diff = std::abs(static_cast<long long>(type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx) - 
                                        type_safe_sub(probe_pair.pt2_rx, probe_pair.pt1_rx)));
    if(adaptive) {
      _threshold = type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx) * threshold;
    }
    good_pos_pairs+=2;
    // fprintf(stderr, "pos pairs: %PRIu64 , %PRIu64 , %PRIu64 , %PRIu64\n", probe_pair.pt1_rx, probe_pair.pt1_tx, probe_pair.pt2_rx, probe_pair.pt2_tx);
    // fprintf(stderr, "pt_tx_delta: %PRIu64 , pt_rx_delta: %PRIu64\n", type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx), type_safe_sub(probe_pair.pt2_rx, probe_pair.pt1_rx));
    if (abs_diff < _threshold) {
      xy_pairs.push_back({static_cast<double>(probe_pair.pt1_tx - min_pt1_tx), 
                          static_cast<double>(signed_sub(probe_pair.pt1_tx, probe_pair.pt1_rx)), 1});
      xy_pairs.push_back({static_cast<double>(probe_pair.pt2_tx - min_pt1_tx), 
                          static_cast<double>(signed_sub(probe_pair.pt2_tx, probe_pair.pt2_rx)), 1});
      filtered_pos_pairs+=2;
    }
    good_neg_pairs+=2;
    abs_diff = std::abs(static_cast<long long>(type_safe_sub(probe_pair.pr2_tx, probe_pair.pr1_tx) - 
                                        type_safe_sub(probe_pair.pr2_rx, probe_pair.pr1_rx)));
    if(adaptive) {
      _threshold = type_safe_sub(probe_pair.pr2_tx, probe_pair.pr1_tx) * threshold;
    }
    // fprintf(stderr, "neg pairs: %PRIu64 , %PRIu64 , %PRIu64 , %PRIu64\n", probe_pair.pr1_rx, probe_pair.pr1_tx, probe_pair.pr2_rx, probe_pair.pr2_tx);
    // fprintf(stderr, "pr_tx_delta: %PRIu64 , pr_rx_delta: %PRIu64\n", type_safe_sub(probe_pair.pr2_tx, probe_pair.pr1_tx), type_safe_sub(probe_pair.pr2_rx, probe_pair.pr1_rx));
    if (abs_diff < _threshold) {
      xy_pairs.push_back({static_cast<double>(probe_pair.pr1_rx - min_pt1_tx), 
                          -static_cast<double>(signed_sub(probe_pair.pr1_tx, probe_pair.pr1_rx)), -1});
      xy_pairs.push_back({static_cast<double>(probe_pair.pr2_rx - min_pt1_tx), 
                          -static_cast<double>(signed_sub(probe_pair.pr2_tx, probe_pair.pr2_rx)), -1});
      filtered_neg_pairs+=2;
    }
  }
  for (const auto& point : xy_pairs) {
    // VLOG(1) << "x: " << point.x << ", y: " << point.y << ", label: " << point.label;
    fprintf(stderr, "x: %f, y: %f, label: %d\n", point.x, point.y, point.label);
  }
  // VLOG(1) << "xy_pairs size: " << xy_pairs.size();
  fprintf(stderr, "xy_pairs size: %zu\n", xy_pairs.size());
  fprintf(stderr, "filtered_pos_pairs/good_pos_pairs: %d/%d, filtered_neg_pairs/good_neg_pairs: %d/%d\n", filtered_pos_pairs, good_pos_pairs, filtered_neg_pairs, good_neg_pairs);
  return xy_pairs;
}

}  // namespace xla::profiler::probe_utils
