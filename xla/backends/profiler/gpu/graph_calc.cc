#include "xla/backends/profiler/gpu/graph_calc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

#include "Eigen/Dense"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/profiler/gpu/probe_data_types.h"

namespace xla::profiler {

namespace {

struct PairHash {
  size_t operator()(const std::pair<int, int>& key) const noexcept {
    return absl::HashOf(key.first, key.second);
  }
};

struct EdgeMeasurement {
  double alpha = 0.0;
  double beta = 0.0;
  int pairs = 0;
  int lost = 0;
  bool synthetic = false;
};

using EdgeKey = std::pair<int, int>;
using EdgeMap = absl::flat_hash_map<EdgeKey, EdgeMeasurement, PairHash>;

std::pair<int, int> MakeUndirected(int a, int b) {
  return (a <= b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

bool IsValidNode(int node_id, int num_nodes) {
  return node_id >= 0 && node_id < num_nodes;
}

uint64_t ComputeMidpoint(uint64_t start_ns, uint64_t end_ns) {
  if (end_ns < start_ns) {
    return start_ns;
  }
  return start_ns + ((end_ns - start_ns) / 2);
}

}  // namespace

GraphCalc::GraphCalc(const Config& config) : config_(config) {
  smoothed_offsets_.assign(config_.num_nodes,
                           std::numeric_limits<double>::quiet_NaN());
}

absl::StatusOr<GraphCalc::RoundResult> GraphCalc::ProcessRound(
    const GlobalWindowData& window) {
  if (config_.num_nodes <= 0) {
    return absl::InvalidArgumentError(
        "GraphCalc requires a positive number of nodes");
  }
  if (config_.reference_node_id < 0 ||
      config_.reference_node_id >= config_.num_nodes) {
    return absl::InvalidArgumentError(
        "GraphCalc reference node is outside configured range");
  }

  RoundResult result;
  result.round_id = window.round_id;
  result.window_id = window.window_id;
  result.window_start_ns.assign(config_.num_nodes, 0);
  result.window_end_ns.assign(config_.num_nodes, 0);
  uint64_t reference_start_ns = 0;
  uint64_t reference_end_ns = 0;
  bool reference_found = false;

  for (const auto& node_data : window.all_nodes) {
    if (node_data.node_id >= 0 && node_data.node_id < config_.num_nodes) {
      result.window_start_ns[node_data.node_id] = node_data.window_start_ns;
      result.window_end_ns[node_data.node_id] = node_data.window_end_ns;
    }
    if (!reference_found &&
        node_data.node_id == config_.reference_node_id) {
      reference_start_ns = node_data.window_start_ns;
      reference_end_ns = node_data.window_end_ns;
      reference_found = true;
    }
  }
  if (!reference_found && !window.all_nodes.empty()) {
    reference_start_ns = window.all_nodes.front().window_start_ns;
    reference_end_ns = window.all_nodes.front().window_end_ns;
  }
  for (int i = 0; i < config_.num_nodes; ++i) {
    if (result.window_start_ns[i] == 0 && reference_found) {
      result.window_start_ns[i] = reference_start_ns;
      result.window_end_ns[i] = reference_end_ns;
    }
  }
  result.midpoint_ns = static_cast<double>(
      ComputeMidpoint(reference_start_ns, reference_end_ns));
  const double reference_midpoint_ns =
      result.midpoint_ns - static_cast<double>(reference_start_ns);

  result.node_offsets.resize(config_.num_nodes);
  for (int i = 0; i < config_.num_nodes; ++i) {
    result.node_offsets[i].node_id = i;
  }

  EdgeMap measurements;
  size_t discarded_edges = 0;
  for (const auto& node_data : window.all_nodes) {
    for (const auto& edge : node_data.edges) {
      if (!IsValidNode(edge.src_node_id, config_.num_nodes) ||
          !IsValidNode(edge.dst_node_id, config_.num_nodes)) {
        ++discarded_edges;
        continue;
      }
      if (edge.pairs_count < config_.min_pairs) {
        ++discarded_edges;
        continue;
      }
      const double loss_ratio =
          edge.pairs_count > 0
              ? static_cast<double>(edge.lost_count) /
                    static_cast<double>(edge.pairs_count)
              : 1.0;
      if (loss_ratio > config_.max_loss_ratio) {
        ++discarded_edges;
        continue;
      }
      if (!std::isfinite(edge.alpha) || !std::isfinite(edge.beta)) {
        ++discarded_edges;
        continue;
      }
      if (std::abs(edge.alpha) > config_.alpha_sanity_bound) {
        ++discarded_edges;
        continue;
      }
      EdgeKey key = {edge.src_node_id, edge.dst_node_id};
      auto& slot = measurements[key];
      if (slot.pairs == 0 || edge.pairs_count > slot.pairs ||
          (!slot.synthetic && edge.pairs_count == slot.pairs)) {
        slot.alpha = edge.alpha;
        slot.beta = edge.beta;
        slot.pairs = edge.pairs_count;
        slot.lost = edge.lost_count;
        slot.synthetic = false;
      }
    }
  }

  if (measurements.empty()) {
    result.failure_reason = "No usable alpha/beta samples for this round";
    return result;
  }

  // Synthesize reverse edges.
  std::vector<EdgeKey> existing_keys;
  existing_keys.reserve(measurements.size());
  for (const auto& entry : measurements) {
    existing_keys.push_back(entry.first);
  }
  for (const auto& key : existing_keys) {
    const auto& measurement = measurements.at(key);
    EdgeKey reverse_key = {key.second, key.first};
    if (measurements.contains(reverse_key)) {
      continue;
    }
    const double denom = 1.0 + measurement.alpha;
    if (std::abs(denom) < config_.reverse_alpha_epsilon) {
      continue;
    }
    EdgeMeasurement reverse;
    reverse.alpha = (-measurement.alpha) / denom;
    reverse.beta = (-measurement.beta) / denom;
    reverse.pairs = measurement.pairs;
    reverse.lost = measurement.lost;
    reverse.synthetic = true;
    measurements.emplace(reverse_key, reverse);
  }

  std::vector<std::vector<int>> adjacency(config_.num_nodes);
  for (const auto& entry : measurements) {
    const EdgeKey& key = entry.first;
    const int src = key.first;
    const int dst = key.second;
    if (!IsValidNode(src, config_.num_nodes) ||
        !IsValidNode(dst, config_.num_nodes)) {
      continue;
    }
    adjacency[src].push_back(dst);
    adjacency[dst].push_back(src);
  }
  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  std::vector<bool> reachable(config_.num_nodes, false);
  std::vector<int> parent(config_.num_nodes, -1);
  std::vector<int> depth(config_.num_nodes, 0);
  std::queue<int> q;
  reachable[config_.reference_node_id] = true;
  parent[config_.reference_node_id] = config_.reference_node_id;
  q.push(config_.reference_node_id);

  while (!q.empty()) {
    int node = q.front();
    q.pop();
    for (int neighbor : adjacency[node]) {
      if (!reachable[neighbor]) {
        reachable[neighbor] = true;
        parent[neighbor] = node;
        depth[neighbor] = depth[node] + 1;
        q.push(neighbor);
      }
    }
  }

  if (!reachable[config_.reference_node_id]) {
    result.failure_reason = "Reference node is not reachable in current graph";
    return result;
  }

  size_t reachable_count =
      std::count(reachable.begin(), reachable.end(), true);
  if (reachable_count <= 1) {
    result.failure_reason =
        "GraphCalc only reached the reference node for this round";
  }

  std::vector<EdgeKey> sorted_edges;
  sorted_edges.reserve(measurements.size());
  for (const auto& entry : measurements) {
    const EdgeKey& key = entry.first;
    if (!reachable[key.first] || !reachable[key.second]) {
      continue;
    }
    sorted_edges.push_back(key);
  }
  std::sort(sorted_edges.begin(), sorted_edges.end());
  sorted_edges.erase(
      std::unique(sorted_edges.begin(), sorted_edges.end()),
      sorted_edges.end());

  if (sorted_edges.empty()) {
    result.failure_reason =
        "No usable edges remained after filtering reachable nodes";
    return result;
  }

  absl::flat_hash_map<EdgeKey, int, PairHash> edge_index;
  edge_index.reserve(sorted_edges.size());
  for (int i = 0; i < sorted_edges.size(); ++i) {
    edge_index.emplace(sorted_edges[i], i);
  }

  std::vector<std::vector<int>> children(config_.num_nodes);
  absl::flat_hash_set<EdgeKey, PairHash> tree_edges;
  for (int node = 0; node < config_.num_nodes; ++node) {
    if (!reachable[node] || node == config_.reference_node_id) {
      continue;
    }
    const int p = parent[node];
    if (p < 0 || !reachable[p]) {
      continue;
    }
    children[p].push_back(node);
    tree_edges.insert(MakeUndirected(node, p));
  }

  Eigen::VectorXd delta_p(sorted_edges.size());
  delta_p.setZero();
  std::vector<double> midpoint(config_.num_nodes,
                               std::numeric_limits<double>::quiet_NaN());

  midpoint[config_.reference_node_id] = reference_midpoint_ns;
  std::queue<int> tree_queue;
  tree_queue.push(config_.reference_node_id);
  while (!tree_queue.empty()) {
    int node = tree_queue.front();
    tree_queue.pop();
    for (int child : children[node]) {
      EdgeKey key = {node, child};
      auto it = measurements.find(key);
      if (it == measurements.end()) {
        continue;
      }
      if (!std::isfinite(midpoint[node])) {
        continue;
      }
      midpoint[child] =
          midpoint[node] + it->second.alpha * midpoint[node] + it->second.beta;
      tree_queue.push(child);
    }
  }

  int valid_offsets = 0;
  for (int node = 0; node < config_.num_nodes; ++node) {
    if (std::isfinite(midpoint[node])) {
      ++valid_offsets;
    }
  }

  for (int idx = 0; idx < sorted_edges.size(); ++idx) {
    const auto& key = sorted_edges[idx];
    const auto& measurement = measurements.at(key);
    const double mp = midpoint[key.first];
    if (!std::isfinite(mp)) {
      delta_p(idx) = 0.0;
    } else {
      delta_p(idx) = measurement.alpha * mp + measurement.beta;
    }
  }

  struct LoopEntry {
    int edge_idx;
    double coeff;
  };
  std::vector<std::vector<LoopEntry>> loops;
  absl::flat_hash_set<EdgeKey, PairHash> processed_loops;
  for (const auto& edge : sorted_edges) {
    auto undirected = MakeUndirected(edge.first, edge.second);
    if (tree_edges.contains(undirected)) {
      continue;
    }
    if (!processed_loops.insert(undirected).second) {
      continue;
    }

    const int src = edge.first;
    const int dst = edge.second;
    if (!reachable[src] || !reachable[dst]) {
      continue;
    }

    // Build tree path from dst back to src.
    std::vector<int> path_from_dst;
    std::vector<int> path_from_src;
    int cursor = dst;
    while (true) {
      path_from_dst.push_back(cursor);
      if (cursor == parent[cursor]) {
        break;
      }
      cursor = parent[cursor];
    }
    cursor = src;
    absl::flat_hash_map<int, int> ancestor_pos;
    for (int i = 0; i < path_from_dst.size(); ++i) {
      ancestor_pos[path_from_dst[i]] = i;
    }
    int lca = -1;
    while (true) {
      path_from_src.push_back(cursor);
      if (ancestor_pos.contains(cursor)) {
        lca = cursor;
        break;
      }
      if (cursor == parent[cursor]) {
        break;
      }
      cursor = parent[cursor];
    }
    if (lca < 0) {
      continue;
    }

    std::vector<LoopEntry> loop_entries;
    // dst path up to LCA.
    int lca_pos = ancestor_pos[lca];
    for (int i = 0; i < lca_pos; ++i) {
      int from = path_from_dst[i];
      int to = path_from_dst[i + 1];
      EdgeKey key = {from, to};
      auto it = edge_index.find(key);
      if (it == edge_index.end()) {
        continue;
      }
      loop_entries.push_back({it->second, 1.0});
    }
    // From LCA down to src.
    for (int i = static_cast<int>(path_from_src.size()) - 1; i > 0; --i) {
      int from = path_from_src[i];
      int to = path_from_src[i - 1];
      EdgeKey key = {from, to};
      auto it = edge_index.find(key);
      if (it == edge_index.end()) {
        continue;
      }
      loop_entries.push_back({it->second, 1.0});
    }
    // Closing edge (src -> dst).
    auto closing = edge_index.find(edge);
    if (closing == edge_index.end()) {
      continue;
    }
    loop_entries.push_back({closing->second, 1.0});

    if (!loop_entries.empty()) {
      loops.push_back(std::move(loop_entries));
    }
  }

  Eigen::VectorXd delta_f = delta_p;
  result.loop_residuals.clear();
  result.loops_constructed = loops.size();
  bool solver_ok = true;
  if (!loops.empty()) {
    Eigen::MatrixXd A(loops.size(), sorted_edges.size());
    A.setZero();
    for (int row = 0; row < loops.size(); ++row) {
      for (const auto& entry : loops[row]) {
        if (entry.edge_idx >= 0 &&
            entry.edge_idx < A.cols()) {
          A(row, entry.edge_idx) += entry.coeff;
        }
      }
    }
    Eigen::VectorXd y = A * delta_p;
    result.loop_residuals.resize(y.size());
    for (int i = 0; i < y.size(); ++i) {
      result.loop_residuals[i] = y(i);
    }
    Eigen::MatrixXd AAt = A * A.transpose();
    Eigen::VectorXd lambda;
    Eigen::LDLT<Eigen::MatrixXd> solver(AAt);
    if (solver.info() == Eigen::Success) {
      lambda = solver.solve(y);
    } else {
      Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(AAt);
      lambda = cod.solve(y);
      solver_ok = (cod.rank() == AAt.rows());
    }
    Eigen::VectorXd correction = A.transpose() * lambda;
    delta_f = delta_p - correction;
  }

  std::queue<int> solve_queue;
  solve_queue.push(config_.reference_node_id);
  while (!solve_queue.empty()) {
    int node = solve_queue.front();
    solve_queue.pop();
    for (int child : children[node]) {
      EdgeKey key = {node, child};
      auto idx_it = edge_index.find(key);
      if (idx_it == edge_index.end()) {
        continue;
      }
      const double parent_mp = midpoint[node];
      if (!std::isfinite(parent_mp)) {
        continue;
      }
      midpoint[child] = parent_mp + delta_f(idx_it->second);
      solve_queue.push(child);
    }
  }

  valid_offsets = 0;
  for (int node = 0; node < config_.num_nodes; ++node) {
    if (std::isfinite(midpoint[node])) {
      ++valid_offsets;
      result.node_offsets[node].reachable = true;
      double raw_offset_ns = midpoint[node] - reference_midpoint_ns;

      if (std::isnan(smoothed_offsets_[node])) {
        smoothed_offsets_[node] = raw_offset_ns;
      } else {
        double gamma = config_.smoothing_factor;
        smoothed_offsets_[node] =
            (1.0 - gamma) * smoothed_offsets_[node] + gamma * raw_offset_ns;
      }

      result.node_offsets[node].offset_ns = smoothed_offsets_[node];
      if (node == config_.reference_node_id) {
        result.node_offsets[node].drift_ppm = 0.0;
        result.node_offsets[node].residual = 0.0;
      } else if (parent[node] >= 0) {
        EdgeKey key = {parent[node], node};
        auto measurement_it = measurements.find(key);
        if (measurement_it != measurements.end()) {
          result.node_offsets[node].drift_ppm =
              measurement_it->second.alpha * 1e6;
        }
        auto idx_it = edge_index.find(key);
        if (idx_it != edge_index.end()) {
          result.node_offsets[node].residual =
              std::abs(delta_p(idx_it->second) - delta_f(idx_it->second));
        }
      }
    }
  }

  result.edges_used = sorted_edges.size();
  result.converged =
      solver_ok && valid_offsets == reachable_count && reachable_count > 0;
  if (!result.converged && result.failure_reason.empty()) {
    result.failure_reason = absl::StrCat(
        "GraphCalc reached ", valid_offsets, " / ", reachable_count,
        " reachable nodes (solver_ok=", solver_ok, ")");
  }

  LOG(INFO) << "GraphCalc round " << result.round_id
            << " edges=" << result.edges_used
            << " loops=" << result.loops_constructed
            << " reachable=" << reachable_count
            << " solved=" << valid_offsets
            << " solver_ok=" << solver_ok;

  if (!result.failure_reason.empty()) {
    VLOG(1) << "GraphCalc diagnostics: " << result.failure_reason
            << " discarded_edges=" << discarded_edges;
  }

  return result;
}

}  // namespace xla::profiler

