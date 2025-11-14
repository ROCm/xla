#ifndef XLA_BACKENDS_PROFILER_GPU_GRAPH_CALC_RUNNER_H_
#define XLA_BACKENDS_PROFILER_GPU_GRAPH_CALC_RUNNER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"

namespace xla::profiler {

class GraphCalcRunner {
 public:
  struct Options {
    std::vector<std::string> input_files;
    std::string output_offsets_path;
    int num_nodes = -1;
    int reference_node = 0;
    int min_pairs = 12;
    double max_loss_ratio = 0.6;
  };

  static absl::Status Run(const Options& options);
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_GRAPH_CALC_RUNNER_H_

