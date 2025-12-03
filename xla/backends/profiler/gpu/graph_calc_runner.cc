#include "xla/backends/profiler/gpu/graph_calc_runner.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "json/json.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/numbers.h"
#include "xla/backends/profiler/gpu/graph_calc.h"
#include "xla/backends/profiler/gpu/probe_data_types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::profiler {

namespace {

using EdgeKey = std::pair<int, int>;

struct EdgeSampleEntry {
  uint64_t round_id = 0;
  EdgeAlphaBeta* edge = nullptr;
  bool missing = false;
};

struct NodeMetadata {
  bool present = false;
  uint64_t start_walltime_ns = 0;
  uint64_t start_gpu_ns = 0;
};

struct AggregatedRound {
  uint64_t round_id = 0;
  uint64_t window_id = 0;
  std::vector<NodeWindowData> nodes;
};

absl::StatusOr<int> InferNodeIdFromFilename(const std::string& path) {
  std::string filename = path;
  size_t slash = filename.find_last_of("/\\");
  if (slash != std::string::npos) {
    filename = filename.substr(slash + 1);
  }
  const std::string marker = "node";
  size_t pos = filename.rfind(marker);
  if (pos == std::string::npos) {
    return absl::NotFoundError(
        absl::StrCat("Could not infer node id from filename ", filename));
  }
  size_t digit_pos = pos + marker.size();
  while (digit_pos < filename.size() &&
         !std::isdigit(static_cast<unsigned char>(filename[digit_pos]))) {
    ++digit_pos;
  }
  if (digit_pos >= filename.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Filename ", filename,
                     " contains \"node\" but no trailing digits"));
  }
  size_t end = digit_pos;
  while (end < filename.size() &&
         std::isdigit(static_cast<unsigned char>(filename[end]))) {
    ++end;
  }
  int node_id = 0;
  if (!absl::SimpleAtoi(filename.substr(digit_pos, end - digit_pos),
                        &node_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unable to parse node id digits in ", filename));
  }
  return node_id;
}

absl::StatusOr<NodeWindowData> ParseNodeWindow(
    const Json::Value& json, const std::optional<int>& default_node_id) {
  if (!json.isObject()) {
    return absl::InvalidArgumentError("Expected JSON object per line");
  }
  NodeWindowData data;
  auto require_uint64 = [&](const char* key,
                            uint64_t* dst) -> absl::Status {
    if (!json.isMember(key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing required field \"", key, "\""));
    }
    const Json::Value& value = json[key];
    if (!value.isUInt64() && !value.isInt64() && !value.isInt()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Field \"", key, "\" must be an integer"));
    }
    *dst = value.asUInt64();
    return absl::OkStatus();
  };

  if (json.isMember("node_id")) {
    uint64_t node_id_uint = 0;
    TF_RETURN_IF_ERROR(require_uint64("node_id", &node_id_uint));
    data.node_id = static_cast<int>(node_id_uint);
  } else if (default_node_id.has_value()) {
    data.node_id = *default_node_id;
  } else {
    return absl::InvalidArgumentError(
        "node_id missing from payload and could not infer from filename");
  }
  TF_RETURN_IF_ERROR(require_uint64("window_id", &data.window_id));
  TF_RETURN_IF_ERROR(require_uint64("round_id", &data.round_id));
  TF_RETURN_IF_ERROR(require_uint64("window_start_ns", &data.window_start_ns));
  TF_RETURN_IF_ERROR(require_uint64("window_end_ns", &data.window_end_ns));

  if (!json.isMember("edges") || !json["edges"].isArray()) {
    return absl::InvalidArgumentError("Field \"edges\" must be an array");
  }
  const Json::Value& edges = json["edges"];
  data.edges.reserve(edges.size());
  for (const auto& edge_json : edges) {
    if (!edge_json.isObject()) {
      return absl::InvalidArgumentError("Edge entries must be objects");
    }
    EdgeAlphaBeta edge;
    edge.src_node_id = data.node_id;
    if (!edge_json.isMember("dst")) {
      return absl::InvalidArgumentError("Edge missing \"dst\"");
    }
    edge.dst_node_id = edge_json["dst"].asInt();
    edge.alpha = edge_json.isMember("alpha") ? edge_json["alpha"].asDouble()
                                             : 0.0;
    edge.beta = edge_json.isMember("beta") ? edge_json["beta"].asDouble() : 0.0;
    edge.pairs_count = edge_json.isMember("pairs")
                           ? edge_json["pairs"].asInt()
                           : 0;
    edge.lost_count =
        edge_json.isMember("lost") ? edge_json["lost"].asInt() : 0;
    data.edges.push_back(edge);
  }
  return data;
}

void SmoothMissingEdges(std::map<uint64_t, AggregatedRound>& aggregated) {
  std::map<EdgeKey, std::vector<EdgeSampleEntry>> history;

  for (auto& [round_id, agg] : aggregated) {
    for (auto& node : agg.nodes) {
      for (auto& edge : node.edges) {
        EdgeKey key{edge.src_node_id, edge.dst_node_id};
        bool missing = (edge.alpha == 0.0 && edge.beta == 0.0);
        history[key].push_back(
            EdgeSampleEntry{round_id, &edge, missing});
      }
    }
  }

  for (auto& [key, samples] : history) {
    if (samples.empty()) {
      continue;
    }
    std::vector<std::optional<std::pair<double, double>>> next_values(
        samples.size());
    std::optional<std::pair<double, double>> next_seen;
    for (int i = static_cast<int>(samples.size()) - 1; i >= 0; --i) {
      auto& entry = samples[i];
      if (!entry.missing) {
        next_seen = std::make_pair(entry.edge->alpha, entry.edge->beta);
      }
      next_values[i] = next_seen;
    }

    std::optional<std::pair<double, double>> prev_seen;
    for (int i = 0; i < samples.size(); ++i) {
      auto& entry = samples[i];
      if (!entry.missing) {
        prev_seen = std::make_pair(entry.edge->alpha, entry.edge->beta);
        continue;
      }
      const auto& next_value = next_values[i];
      std::optional<std::pair<double, double>> chosen;
      if (prev_seen && next_value) {
        chosen = std::make_pair((prev_seen->first + next_value->first) / 2.0,
                                (prev_seen->second + next_value->second) / 2.0);
      } else if (prev_seen) {
        chosen = prev_seen;
      } else if (next_value) {
        chosen = next_value;
      }
      if (chosen) {
        entry.edge->alpha = chosen->first;
        entry.edge->beta = chosen->second;
        entry.missing = false;
      }
    }
  }
}

absl::StatusOr<std::vector<std::string>> ReadLines(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open file: ", path));
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

absl::Status ParseFile(const std::string& path,
                       std::map<uint64_t, AggregatedRound>* rounds,
                       int* max_node_id,
                       std::map<int, NodeMetadata>* metadata) {
  std::optional<int> inferred_node_id;
  auto inferred = InferNodeIdFromFilename(path);
  if (inferred.ok()) {
    inferred_node_id = *inferred;
  } else {
    LOG(WARNING) << "Failed to infer node id from filename " << path
                 << ": " << inferred.status();
  }
  TF_ASSIGN_OR_RETURN(auto lines, ReadLines(path));
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  for (const std::string& line : lines) {
    Json::Value json_root;
    std::string errors;
    if (!reader->parse(line.data(), line.data() + line.size(), &json_root,
                       &errors)) {
      return absl::InvalidArgumentError(
          absl::StrCat("JSON parse error in ", path, ": ", errors));
    }
    if (json_root.isMember("meta") && json_root["meta"].asBool()) {
      if (!json_root.isMember("node_id")) {
        return absl::InvalidArgumentError(
            "Metadata line missing node_id");
      }
      int meta_node = json_root["node_id"].asInt();
      NodeMetadata& meta_entry = (*metadata)[meta_node];
      meta_entry.present = true;
      if (json_root.isMember("start_walltime_ns")) {
        meta_entry.start_walltime_ns =
            json_root["start_walltime_ns"].asUInt64();
      }
      if (json_root.isMember("start_gpu_ns")) {
        meta_entry.start_gpu_ns = json_root["start_gpu_ns"].asUInt64();
      }
      *max_node_id = std::max(*max_node_id, meta_node);
      continue;
    }

    TF_ASSIGN_OR_RETURN(NodeWindowData node_window,
                        ParseNodeWindow(json_root, inferred_node_id));
    AggregatedRound& agg = (*rounds)[node_window.round_id];
    agg.round_id = node_window.round_id;
    agg.window_id = std::max(agg.window_id, node_window.window_id);
    agg.nodes.push_back(std::move(node_window));
    auto& stored = agg.nodes.back();
    *max_node_id = std::max(*max_node_id, stored.node_id);
    for (const auto& edge : stored.edges) {
      *max_node_id = std::max(*max_node_id, edge.dst_node_id);
    }
  }
  LOG(INFO) << "Loaded " << lines.size() << " rows from " << path;
  return absl::OkStatus();
}

absl::Status WriteOffsets(const std::string& path,
                          const std::vector<GraphCalc::RoundResult>& rounds,
                          const std::vector<NodeMetadata>& metadata) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open output file ", path));
  }
  if (!metadata.empty()) {
    out << "{\"meta\":true,\"nodes\":[";
    bool first = true;
    for (int i = 0; i < metadata.size(); ++i) {
      if (!first) {
        out << ",";
      }
      first = false;
      out << "{\"node_id\":" << i
          << ",\"start_walltime_ns\":" << metadata[i].start_walltime_ns
          << ",\"start_gpu_ns\":" << metadata[i].start_gpu_ns
          << "}";
    }
    out << "]}\n";
  }
  for (const auto& round : rounds) {
    for (const auto& node : round.node_offsets) {
      uint64_t node_start_ns =
          (node.node_id >= 0 &&
           node.node_id < round.window_start_ns.size())
              ? round.window_start_ns[node.node_id]
              : 0;
      uint64_t node_end_ns =
          (node.node_id >= 0 &&
           node.node_id < round.window_end_ns.size())
              ? round.window_end_ns[node.node_id]
              : 0;
      out << "{\"round_id\":" << round.round_id
          << ",\"window_id\":" << round.window_id
          << ",\"window_start_ns\":" << node_start_ns
          << ",\"window_end_ns\":" << node_end_ns
          << ",\"node_id\":" << node.node_id
          << ",\"offset_ns\":" << node.offset_ns
          << ",\"drift_ppm\":" << node.drift_ppm
          << ",\"residual\":" << node.residual
          << ",\"reachable\":" << (node.reachable ? "true" : "false")
          << ",\"converged\":" << (round.converged ? "true" : "false")
          << "}\n";
    }
  }
  out.flush();
  LOG(INFO) << "Wrote " << rounds.size() << " rounds to " << path;
  return absl::OkStatus();
}

}  // namespace

absl::Status GraphCalcRunner::Run(const Options& options) {
  if (options.input_files.empty()) {
    return absl::InvalidArgumentError(
        "GraphCalcRunner requires at least one input file");
  }
  if (options.reference_node < 0) {
    return absl::InvalidArgumentError("reference_node must be non-negative");
  }
  std::map<uint64_t, AggregatedRound> aggregated;
  int max_node_id = -1;
  std::map<int, NodeMetadata> metadata_map;
  for (const auto& file : options.input_files) {
    TF_RETURN_IF_ERROR(
        ParseFile(file, &aggregated, &max_node_id, &metadata_map));
  }

  if (aggregated.empty()) {
    return absl::InvalidArgumentError(
        "No rounds were parsed from the provided files");
  }

  int num_nodes = options.num_nodes;
  if (num_nodes <= 0) {
    num_nodes = max_node_id + 1;
  }
  if (num_nodes <= 0) {
    return absl::InvalidArgumentError(
        "Unable to infer num_nodes from input; please pass --num_nodes");
  }
  if (options.reference_node >= num_nodes) {
    return absl::InvalidArgumentError(
        "reference_node must be < num_nodes");
  }
  SmoothMissingEdges(aggregated);
  GraphCalc::Config calc_config;
  calc_config.reference_node_id = options.reference_node;
  calc_config.num_nodes = num_nodes;
  calc_config.min_pairs = options.min_pairs;
  calc_config.max_loss_ratio = options.max_loss_ratio;
  calc_config.smoothing_factor = options.smoothing_factor;

  std::vector<NodeMetadata> metadata(num_nodes);
  for (const auto& [node_id, meta] : metadata_map) {
    if (node_id >= 0 && node_id < num_nodes) {
      metadata[node_id] = meta;
    }
  }

  GraphCalc calc(calc_config);
  std::vector<GraphCalc::RoundResult> output_rounds;
  output_rounds.reserve(aggregated.size());
  for (auto& [round_id, agg] : aggregated) {
    GlobalWindowData global;
    global.round_id = round_id;
    global.window_id = agg.window_id;
    global.window_start_ns.assign(num_nodes, 0);
    global.window_end_ns.assign(num_nodes, 0);
    global.all_nodes = std::move(agg.nodes);
    for (const auto& node_data : global.all_nodes) {
      if (node_data.node_id >= 0 && node_data.node_id < num_nodes) {
        global.window_start_ns[node_data.node_id] =
            node_data.window_start_ns;
        global.window_end_ns[node_data.node_id] =
            node_data.window_end_ns;
      }
    }
    auto calc_result = calc.ProcessRound(global);
    if (!calc_result.ok()) {
      LOG(WARNING) << "GraphCalc failed for round " << round_id << ": "
                   << calc_result.status();
      continue;
    }
    output_rounds.push_back(*std::move(calc_result));
  }

  if (output_rounds.empty()) {
    return absl::InternalError("GraphCalc did not produce any results");
  }

  std::string output_path = options.output_offsets_path;
  if (output_path.empty()) {
    output_path = "round_offsets.jsonl";
  }
  TF_RETURN_IF_ERROR(WriteOffsets(output_path, output_rounds, metadata));
  return absl::OkStatus();
}

}  // namespace xla::profiler

