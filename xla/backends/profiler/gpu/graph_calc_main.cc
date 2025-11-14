#include "xla/backends/profiler/gpu/graph_calc_runner.h"

#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"

ABSL_FLAG(std::vector<std::string>, input_files, {},
          "List of per-node JSONL files to ingest.");
ABSL_FLAG(std::string, output_offsets, "round_offsets.jsonl",
          "Destination JSONL file for per-node offsets.");
ABSL_FLAG(int, num_nodes, -1,
          "Total number of nodes participating in the probing graph. "
          "If unspecified, inferred from the input.");
ABSL_FLAG(int, reference_node, 0,
          "Reference node id whose offset remains zero.");
ABSL_FLAG(int, min_pairs, 12,
          "Minimum probe pairs per edge required to participate.");
ABSL_FLAG(double, max_loss_ratio, 0.6,
          "Maximum allowed loss ratio for an edge to be considered.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  xla::profiler::GraphCalcRunner::Options options;
  options.input_files = absl::GetFlag(FLAGS_input_files);
  options.output_offsets_path = absl::GetFlag(FLAGS_output_offsets);
  options.num_nodes = absl::GetFlag(FLAGS_num_nodes);
  options.reference_node = absl::GetFlag(FLAGS_reference_node);
  options.min_pairs = absl::GetFlag(FLAGS_min_pairs);
  options.max_loss_ratio = absl::GetFlag(FLAGS_max_loss_ratio);

  absl::Status status = xla::profiler::GraphCalcRunner::Run(options);
  if (!status.ok()) {
    LOG(ERROR) << "GraphCalcRunner failed: " << status;
    return 1;
  }
  LOG(INFO) << "GraphCalcRunner completed successfully";
  return 0;
}

