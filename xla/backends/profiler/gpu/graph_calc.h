#ifndef XLA_BACKENDS_PROFILER_GPU_GRAPH_CALC_H_
#define XLA_BACKENDS_PROFILER_GPU_GRAPH_CALC_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/profiler/gpu/probe_data_types.h"

namespace xla::profiler {

class GraphCalc {
 public:
  struct Config {
    int reference_node_id = 0;
    int num_nodes = 1;
    int min_pairs = 12;
    double max_loss_ratio = 0.5;
    double alpha_sanity_bound = 1.0;
    double reverse_alpha_epsilon = 1e-6;
    double smoothing_factor = 1.0;
  };

  struct NodeOffset {
    int node_id = -1;
    double offset_ns = 0.0;
    double drift_ppm = 0.0;
    double residual = 0.0;
    bool reachable = false;
  };

  struct RoundResult {
    uint64_t round_id = 0;
    uint64_t window_id = 0;
    std::vector<uint64_t> window_start_ns;
    std::vector<uint64_t> window_end_ns;
    double midpoint_ns = 0.0;
    std::vector<NodeOffset> node_offsets;
    std::vector<double> loop_residuals;
    size_t edges_used = 0;
    size_t loops_constructed = 0;
    bool converged = false;
    std::string failure_reason;
  };

  explicit GraphCalc(const Config& config);

  absl::StatusOr<RoundResult> ProcessRound(const GlobalWindowData& window);

 private:
  Config config_;
  std::vector<double> smoothed_offsets_;
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_GRAPH_CALC_H_

