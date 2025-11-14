#ifndef XLA_BACKENDS_PROFILER_GPU_PROBE_DATA_TYPES_H_
#define XLA_BACKENDS_PROFILER_GPU_PROBE_DATA_TYPES_H_

#include <cstdint>
#include <vector>

namespace xla::profiler {

struct EdgeAlphaBeta {
  int src_node_id = -1;
  int dst_node_id = -1;
  double alpha = 0.0;
  double beta = 0.0;
  int pairs_count = 0;
  int lost_count = 0;
};

struct NodeWindowData {
  int node_id = -1;
  uint64_t window_id = 0;
  uint64_t round_id = 0;
  uint64_t window_start_ns = 0;
  uint64_t window_end_ns = 0;
  std::vector<EdgeAlphaBeta> edges;
};

struct GlobalWindowData {
  uint64_t window_id = 0;
  uint64_t round_id = 0;
  std::vector<uint64_t> window_start_ns;
  std::vector<uint64_t> window_end_ns;
  std::vector<NodeWindowData> all_nodes;
  uint64_t sequence_number = 0;
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_PROBE_DATA_TYPES_H_

