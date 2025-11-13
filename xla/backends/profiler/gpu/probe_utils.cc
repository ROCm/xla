#include "xla/backends/profiler/gpu/probe_utils.h"

#include <cstdio>
#include <algorithm>
#include "absl/container/flat_hash_map.h"
#include <cmath>
#include "absl/log/log.h"
#include "svm_wrapper.h"

namespace xla::profiler::probe_utils {

ProbInfo convert_probe_pairs_to_xy_pairs(
    const absl::flat_hash_map<uint32_t, probe_info::ProbePair>& probe_pairs, 
    double threshold,
    bool adaptive
) {
  ProbInfo prob_info;
  std::vector<Point> xy_pairs;
  uint64_t min_pt1_tx = 0, max_pr2_rx = 0;
  uint64_t x_min = 0, x_max = 0;
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
    if(x_max == 0 || probe_pair.pt1_tx < x_max) {
      x_min = probe_pair.pt1_tx;
    }
    x_max = std::max(x_max, probe_pair.pr2_rx);
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
  auto std_diff = static_cast<double>(signed_sub(x_min, x_max));
  for (const auto& [seq_id, probe_pair] : probe_pairs) {
    if(probe_pair.pt1_tx == 0 || probe_pair.pt2_tx == 0 || probe_pair.pt1_rx == 0 || probe_pair.pt2_rx == 0) {
      continue;
    }
    if(probe_pair.pr1_tx == 0 || probe_pair.pr2_tx == 0 || probe_pair.pr1_rx == 0 || probe_pair.pr2_rx == 0) {
      continue;
    }
    
    auto abs_diff = std::abs(static_cast<long long>(type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx) - 
                                        type_safe_sub(probe_pair.pt2_rx, probe_pair.pt1_rx)));
    
    if(adaptive) {
      _threshold = type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx) * threshold;
    }
    good_pos_pairs+=2;
    // fprintf(stderr, "pos pairs: %PRIu64 , %PRIu64 , %PRIu64 , %PRIu64\n", probe_pair.pt1_rx, probe_pair.pt1_tx, probe_pair.pt2_rx, probe_pair.pt2_tx);
    // fprintf(stderr, "pt_tx_delta: %PRIu64 , pt_rx_delta: %PRIu64\n", type_safe_sub(probe_pair.pt2_tx, probe_pair.pt1_tx), type_safe_sub(probe_pair.pt2_rx, probe_pair.pt1_rx));
    if (abs_diff < _threshold) {
      xy_pairs.push_back({static_cast<double>(signed_sub(x_min, probe_pair.pt1_tx)) / std_diff, 
                          static_cast<double>(signed_sub(probe_pair.pt1_tx, probe_pair.pt1_rx)), 1});
      xy_pairs.push_back({static_cast<double>(signed_sub(x_min, probe_pair.pt2_tx)) / std_diff, 
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
      xy_pairs.push_back({static_cast<double>(signed_sub(x_min, probe_pair.pr1_rx)) / std_diff, 
                          -static_cast<double>(signed_sub(probe_pair.pr1_tx, probe_pair.pr1_rx)), -1});
      xy_pairs.push_back({static_cast<double>(signed_sub(x_min, probe_pair.pr2_rx)) / std_diff, 
                          -static_cast<double>(signed_sub(probe_pair.pr2_tx, probe_pair.pr2_rx)), -1});
      filtered_neg_pairs+=2;
    }
  }
  double y_max = -1e14, y_min = 1e14;
  for (const auto& point : xy_pairs) {
    y_max = std::max(y_max, point.y);
    y_min = std::min(y_min, point.y);
  }
  for (auto& point : xy_pairs) {
    // continue;
    point.y = (point.y - y_min) / (y_max - y_min);
  }
  ScaleInfo scale_info;
  scale_info.y_max = static_cast<double>(y_max);
  scale_info.y_min = static_cast<double>(y_min);
  scale_info.x_max = static_cast<double>(x_max);
  scale_info.x_min = static_cast<double>(x_min);
  prob_info.scale_info = scale_info;
  prob_info.points = xy_pairs;
  fprintf(stderr, "y_max: %f, y_min: %f\n", y_max, y_min);
  fprintf(stderr, "x_diff: %lf\n", std_diff);
  for (const auto& point : xy_pairs) {
    // VLOG(1) << "x: " << point.x << ", y: " << point.y << ", label: " << point.label;
    fprintf(stderr, "x: %f, y: %f, label: %d\n", point.x, point.y, point.label);
  }
  // VLOG(1) << "xy_pairs size: " << xy_pairs.size();
  fprintf(stderr, "xy_pairs size: %zu\n", xy_pairs.size());
  fprintf(stderr, "filtered_pos_pairs/good_pos_pairs: %d/%d, filtered_neg_pairs/good_neg_pairs: %d/%d\n", filtered_pos_pairs, good_pos_pairs, filtered_neg_pairs, good_neg_pairs);
  // return xy_pairs;
  return prob_info;
}

}  // namespace xla::profiler::probe_utils
