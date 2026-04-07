/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/rocm/rocm_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<hipGraph_t> CreateGraph() {
  VLOG(2) << "Create new HIP graph";
  hipGraph_t graph;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphCreate(&graph, /*flags=*/0),
                              "Failed to create HIP graph"));
  VLOG(2) << "Created HIP graph " << graph;
  return graph;
}

hipDeviceptr_t AsDevicePtr(const DeviceAddressBase& mem) {
  return absl::bit_cast<hipDeviceptr_t>(mem.opaque());
}

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;

// Converts a platform independent GraphNodeHandle into a HIP specific
// hipGraphNode_t.
hipGraphNode_t ToHipGraphHandle(GpuCommandBuffer::GraphNodeHandle handle) {
  return absl::bit_cast<hipGraphNode_t>(handle);
}

// Converts a list of platform independent GraphNodeHandles into a list of
// HIP specific hipGraphNode_t.
std::vector<hipGraphNode_t> ToHipGraphHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<hipGraphNode_t> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToHipGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a HIP specific hipGraphNode_t into a platform independent
// GraphNodeHandle. This function will be removed once all Node factory
// functions have been migrated into the subclasses.
GraphNodeHandle FromHipGraphHandle(hipGraphNode_t handle) {
  return absl::bit_cast<GpuCommandBuffer::GraphNodeHandle>(handle);
}
}  // namespace

absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> RocmCommandBuffer::Create(
    Mode mode, StreamExecutor* executor) {
  TF_ASSIGN_OR_RETURN(hipGraph_t graph, CreateGraph());
  return std::unique_ptr<RocmCommandBuffer>(
      new RocmCommandBuffer(mode, executor, graph, /*is_owned_graph=*/true));
}

absl::StatusOr<GpuCommandBuffer::GraphConditionalNodeHandle>
RocmCommandBuffer::CreateConditionalNode(
    absl::Span<const GraphNodeHandle> dependencies,
    GraphConditionalHandle conditional, ConditionType type) {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateSetCaseConditionNode(
    absl::Span<const GraphConditionalHandle> conditionals,
    DeviceAddress<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    bool enable_conditional_default,
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::Status RocmCommandBuffer::UpdateSetCaseConditionNode(
    GraphNodeHandle handle,
    absl::Span<const GraphConditionalHandle> conditionals,
    DeviceAddress<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    bool enable_conditional_default) {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateSetWhileConditionNode(
    GraphConditionalHandle conditional, DeviceAddress<bool> predicate,
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::Status RocmCommandBuffer::UpdateSetWhileConditionNode(
    GraphNodeHandle handle, GraphConditionalHandle conditional,
    DeviceAddress<bool> predicate) {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateMemsetNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceAddressBase destination, BitPattern bit_pattern,
    size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; deps: " << dependencies.size();

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddMemsetNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add memset node to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
                                                 DeviceAddressBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements;

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return ToStatus(wrap::hipGraphExecMemsetNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), &params),
                  "Failed to set memset node params");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateMemcpyD2DNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceAddressBase destination, DeviceAddressBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; deps: " << dependencies.size();

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipGraphAddMemcpyNode1D(&node_handle, graph_, deps.data(),
                                    deps.size(), AsDevicePtr(destination),
                                    AsDevicePtr(source), size,
                                    hipMemcpyDeviceToDevice),
      "Failed to add memcpy d2d node to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateMemcpyD2DNode(
    GraphNodeHandle node_handle, DeviceAddressBase destination,
    DeviceAddressBase source, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node_handle
          << " in graph executable " << exec_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size;

  return ToStatus(
      wrap::hipGraphExecMemcpyNodeSetParams1D(
          exec_, ToHipGraphHandle(node_handle), AsDevicePtr(destination),
          AsDevicePtr(source), size, hipMemcpyDeviceToDevice),
      "Failed to set memcpy d2d node params");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateClonedChildNode(
    absl::Span<const GraphNodeHandle> dependencies,
    const CommandBuffer& nested) {
  auto& child_command_buffer = tensorflow::down_cast<RocmCommandBuffer&>(
      const_cast<CommandBuffer&>(nested));
  CHECK(child_command_buffer.parent_ == nullptr)
      << "Nested command buffer's parent is not null";
  child_command_buffer.parent_ = this;
  hipGraph_t child_graph = child_command_buffer.graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_ << "; deps: " << dependencies.size();

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipGraphAddChildGraphNode(&node_handle, graph_, deps.data(),
                                      deps.size(), child_graph),
      "Failed to create a child graph node and add it to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateMovedChildNode(
    absl::Span<const GraphNodeHandle> dependencies, CommandBuffer* nested) {
  return absl::UnimplementedError("Moved child nodes are not supported");
}

absl::Status RocmCommandBuffer::UpdateClonedChildNode(
    GraphNodeHandle node_handle, const CommandBuffer& nested) {
  hipGraph_t child_graph =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested).graph_;

  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << exec_ << "to params contained in " << child_graph;

  return ToStatus(wrap::hipGraphExecChildGraphNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), child_graph),
                  "Failed to set HIP graph child node params");
}

absl::StatusOr<const CommandBuffer::Command*>
RocmCommandBuffer::FlattenChildGraphNodes(
    const CommandBuffer& nested,
    absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  const auto& child_buf =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested);
  hipGraph_t child_graph = child_buf.graph_;

  size_t num_nodes = 0;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(child_graph, nullptr, &num_nodes),
               "Failed to get child graph node count"));

  if (num_nodes == 0) {
    GpuFlattenedCommand empty_cmd;
    return AppendCommand(std::move(empty_cmd));
  }

  std::vector<hipGraphNode_t> child_nodes(num_nodes);
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(child_graph, child_nodes.data(),
                                      &num_nodes),
               "Failed to enumerate child graph nodes"));

  size_t num_edges = 0;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetEdges(child_graph, nullptr, nullptr,
                                      &num_edges),
               "Failed to get child graph edge count"));

  absl::flat_hash_map<hipGraphNode_t, int> in_degree;
  absl::flat_hash_map<hipGraphNode_t, std::vector<hipGraphNode_t>> successors;
  for (auto& n : child_nodes) in_degree[n] = 0;

  if (num_edges > 0) {
    std::vector<hipGraphNode_t> from(num_edges), to(num_edges);
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphGetEdges(child_graph, from.data(), to.data(),
                                        &num_edges),
                 "Failed to enumerate child graph edges"));
    for (size_t i = 0; i < num_edges; ++i) {
      in_degree[to[i]]++;
      successors[from[i]].push_back(to[i]);
    }
  }

  // Topological sort (Kahn's algorithm).
  std::vector<hipGraphNode_t> sorted;
  sorted.reserve(num_nodes);
  {
    std::vector<hipGraphNode_t> q;
    for (auto& n : child_nodes) {
      if (in_degree[n] == 0) q.push_back(n);
    }
    while (!q.empty()) {
      hipGraphNode_t n = q.back();
      q.pop_back();
      sorted.push_back(n);
      for (auto& s : successors[n]) {
        if (--in_degree[s] == 0) q.push_back(s);
      }
    }
  }

  VLOG(1) << "FlattenChildGraphNodes: " << num_nodes << " nodes, "
          << num_edges << " edges from child graph " << child_graph;

  absl::flat_hash_map<hipGraphNode_t, GraphNodeHandle> child_to_parent;
  std::vector<GraphNodeHandle> dep_handles;
  for (auto* d : dependencies) {
    auto* gpu_cmd = static_cast<const GpuCommand*>(d);
    dep_handles.push_back(gpu_cmd->handle);
  }

  GpuFlattenedCommand flat_cmd;

  for (hipGraphNode_t child_node : sorted) {
    hipGraphNodeType type;
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphNodeGetType(child_node, &type),
                 "Failed to get node type"));

    // Resolve dependencies: use predecessor parent handles or external deps.
    std::vector<GraphNodeHandle> node_deps;
    size_t n_preds = 0;
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphNodeGetDependencies(child_node, nullptr,
                                                    &n_preds),
                 "Failed to get node dependencies count"));
    bool has_child_preds = false;
    if (n_preds > 0) {
      std::vector<hipGraphNode_t> preds(n_preds);
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphNodeGetDependencies(child_node, preds.data(),
                                                      &n_preds),
                   "Failed to get node dependencies"));
      for (auto& p : preds) {
        auto it = child_to_parent.find(p);
        if (it != child_to_parent.end()) {
          node_deps.push_back(it->second);
          has_child_preds = true;
        }
      }
    }
    if (!has_child_preds) node_deps = dep_handles;

    std::vector<hipGraphNode_t> hip_deps;
    hip_deps.reserve(node_deps.size());
    for (auto& h : node_deps) hip_deps.push_back(ToHipGraphHandle(h));

    hipGraphNode_t new_node = nullptr;

    if (type == hipGraphNodeTypeKernel) {
      hipKernelNodeParams kparams;
      memset(&kparams, 0, sizeof(kparams));
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphKernelNodeGetParams(child_node, &kparams),
                   "Failed to get kernel node params"));

      if (kparams.sharedMemBytes != 0 && kparams.func != nullptr) {
        TF_RETURN_IF_ERROR(ToStatus(
            wrap::hipFuncSetAttribute(
                kparams.func,
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kparams.sharedMemBytes),
            "Failed to set shared memory size for flattened kernel"));
      }

      VLOG(2) << "FlattenChildGraphNodes: kernel func=" << kparams.func
              << " grid=(" << kparams.gridDim.x << ","
              << kparams.gridDim.y << "," << kparams.gridDim.z
              << ") block=(" << kparams.blockDim.x << ","
              << kparams.blockDim.y << "," << kparams.blockDim.z
              << ") shmem=" << kparams.sharedMemBytes
              << " kernelParams=" << kparams.kernelParams
              << " extra=" << kparams.extra;

      hipError_t kerr = wrap::hipGraphAddKernelNode(
          &new_node, graph_, hip_deps.data(), hip_deps.size(), &kparams);
      if (kerr != hipSuccess) {
        // Kernels using opaque `extra` arg-packing can't be replicated.
        // Abort flattening and let the caller fall back to child graph.
        VLOG(1) << "FlattenChildGraphNodes: kernel with extra args can't "
                   "be flattened (error=" << kerr << "), aborting";
        return absl::UnimplementedError(
            "Cannot flatten kernel with opaque extra args");
      }

    } else if (type == hipGraphNodeTypeMemcpy) {
      hipMemcpy3DParms mparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemcpyNodeGetParams(child_node, &mparams),
                   "Failed to get memcpy node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphAddMemcpyNode(&new_node, graph_, hip_deps.data(),
                                     hip_deps.size(), &mparams),
          "Failed to add flattened memcpy node"));

    } else if (type == hipGraphNodeTypeMemset) {
      hipMemsetParams msparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemsetNodeGetParams(child_node, &msparams),
                   "Failed to get memset node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphAddMemsetNode(&new_node, graph_, hip_deps.data(),
                                     hip_deps.size(), &msparams),
          "Failed to add flattened memset node"));

    } else if (type == hipGraphNodeTypeEmpty) {
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphAddEmptyNode(&new_node, graph_, hip_deps.data(),
                                    hip_deps.size()),
          "Failed to add flattened empty node"));
    } else {
      VLOG(1) << "FlattenChildGraphNodes: skipping node type "
              << static_cast<int>(type);
      continue;
    }

    GraphNodeHandle parent_handle = FromHipGraphHandle(new_node);
    child_to_parent[child_node] = parent_handle;
    flat_cmd.node_handles.push_back(parent_handle);
  }

  VLOG(1) << "FlattenChildGraphNodes: created " << flat_cmd.node_handles.size()
          << " nodes in parent graph " << graph_;
  return AppendCommand(std::move(flat_cmd));
}

absl::Status RocmCommandBuffer::UpdateFlattenedChildNodes(
    const Command* command, const CommandBuffer& nested) {
  auto* flat_cmd = static_cast<const GpuFlattenedCommand*>(command);
  const auto& child_buf =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested);
  hipGraph_t child_graph = child_buf.graph_;

  // Get the new child graph's nodes in the same topological order.
  size_t num_nodes = 0;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(child_graph, nullptr, &num_nodes),
               "Failed to get child graph node count"));

  std::vector<hipGraphNode_t> child_nodes(num_nodes);
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(child_graph, child_nodes.data(),
                                      &num_nodes),
               "Failed to enumerate child graph nodes"));

  // Topological sort.
  size_t num_edges = 0;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetEdges(child_graph, nullptr, nullptr,
                                      &num_edges),
               "Failed to get child graph edge count"));

  absl::flat_hash_map<hipGraphNode_t, int> in_degree;
  absl::flat_hash_map<hipGraphNode_t, std::vector<hipGraphNode_t>> successors;
  for (auto& n : child_nodes) in_degree[n] = 0;
  if (num_edges > 0) {
    std::vector<hipGraphNode_t> from(num_edges), to(num_edges);
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphGetEdges(child_graph, from.data(), to.data(),
                                        &num_edges),
                 "Failed to enumerate child graph edges"));
    for (size_t i = 0; i < num_edges; ++i) {
      in_degree[to[i]]++;
      successors[from[i]].push_back(to[i]);
    }
  }

  std::vector<hipGraphNode_t> sorted;
  sorted.reserve(num_nodes);
  {
    std::vector<hipGraphNode_t> q;
    for (auto& n : child_nodes) {
      if (in_degree[n] == 0) q.push_back(n);
    }
    while (!q.empty()) {
      hipGraphNode_t n = q.back();
      q.pop_back();
      sorted.push_back(n);
      for (auto& s : successors[n]) {
        if (--in_degree[s] == 0) q.push_back(s);
      }
    }
  }

  // Update each flattened node with new params from the corresponding child
  // graph node.  The topology must be identical.
  size_t flat_idx = 0;
  for (hipGraphNode_t child_node : sorted) {
    if (flat_idx >= flat_cmd->node_handles.size()) break;

    hipGraphNodeType type;
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphNodeGetType(child_node, &type),
                 "Failed to get node type"));

    GraphNodeHandle parent_handle = flat_cmd->node_handles[flat_idx];

    if (type == hipGraphNodeTypeKernel) {
      hipKernelNodeParams kparams;
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphKernelNodeGetParams(child_node, &kparams),
                   "Failed to get kernel node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphExecKernelNodeSetParams(
              exec_, ToHipGraphHandle(parent_handle), &kparams),
          "Failed to update flattened kernel node"));

    } else if (type == hipGraphNodeTypeMemcpy) {
      hipMemcpy3DParms mparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemcpyNodeGetParams(child_node, &mparams),
                   "Failed to get memcpy node params"));
      // Use 1D shortcut when possible: extract src, dst, size from 3D params.
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphExecMemcpyNodeSetParams1D(
              exec_, ToHipGraphHandle(parent_handle),
              mparams.dstArray ? mparams.dstPtr.ptr : mparams.dstPtr.ptr,
              mparams.srcArray ? mparams.srcPtr.ptr : mparams.srcPtr.ptr,
              mparams.extent.width, hipMemcpyDeviceToDevice),
          "Failed to update flattened memcpy node"));

    } else if (type == hipGraphNodeTypeMemset) {
      hipMemsetParams msparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemsetNodeGetParams(child_node, &msparams),
                   "Failed to get memset node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphExecMemsetNodeSetParams(
              exec_, ToHipGraphHandle(parent_handle), &msparams),
          "Failed to update flattened memset node"));
    }
    // Empty nodes need no update.

    ++flat_idx;
  }

  VLOG(3) << "UpdateFlattenedChildNodes: updated " << flat_idx
          << " nodes in exec " << exec_;
  return absl::OkStatus();
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateKernelNode(
    absl::Span<const GraphNodeHandle> dependencies, StreamPriority priority,
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes
          << "; deps: " << dependencies.size();

  hipKernelNodeParams params{};

  hipFunction_t function =
      static_cast<const RocmKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.y = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add kernel node to a HIP graph"));

  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateKernelNode(
    GraphNodeHandle node_handle, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << node_handle << " in graph executable "
          << exec_ << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  hipKernelNodeParams params{};

  hipFunction_t function =
      static_cast<const RocmKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.y = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return ToStatus(wrap::hipGraphExecKernelNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), &params),
                  "Failed to set HIP graph kernel node params");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateEmptyNode(
    absl::Span<const GraphNodeHandle> dependencies) {
  VLOG(2) << "Add empty node to a graph " << graph_
          << "; deps: " << dependencies.size();

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddEmptyNode(&node_handle, graph_, deps.data(),
                                          deps.size()),
               "Failed to add empty node to a HIP graph"));

  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());
  TF_ASSIGN_OR_RETURN(size_t count, GetNodeCount());
  if (count != 0 || !is_owned_graph_)
    return absl::InternalError(
        "Stream can't be traced on non empty command buffer");

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  hipStream_t stream_handle =
      static_cast<hipStream_t>(stream->platform_specific_handle().stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamBeginCapture(stream_handle,
                                           hipStreamCaptureModeThreadLocal),
               "Failed to begin stream capture"));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  VLOG(5) << "End stream " << stream << " capture";
  hipGraph_t captured_graph;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamEndCapture(stream_handle, &captured_graph),
               "Failed to end stream capture"));
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphDestroy(std::exchange(graph_, captured_graph)),
               "Failed to destroy HIP graph"));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

absl::Status RocmCommandBuffer::LaunchGraph(Stream* stream) {
  VLOG(3) << "Launch command buffer executable graph " << exec_
          << " on a stream: " << stream;
  return ToStatus(wrap::hipGraphLaunch(
                      exec_, static_cast<hipStream_t>(
                                 stream->platform_specific_handle().stream)),
                  "Failed to launch HIP graph");
}
absl::StatusOr<size_t> RocmCommandBuffer::GetNodeCount() const {
  size_t numNodes;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(graph_, /*nodes=*/nullptr, &numNodes),
               "Failed to get HIP graph node count"));

  return numNodes;
}

absl::Status RocmCommandBuffer::PrepareFinalization() {
  return absl::OkStatus();
}

absl::StatusOr<GpuCommandBuffer::GraphConditionalHandle>
RocmCommandBuffer::CreateConditionalHandle() {
  return absl::UnimplementedError(
      "Graph conditionals are not yet supported on HIP graphs.");
}

absl::Status RocmCommandBuffer::WriteGraphToDotFile(absl::string_view path) {
  VLOG(2) << "Print HIP graph " << graph_ << " debug dot file to " << path;

  int flags = hipGraphDebugDotFlagsVerbose;
  return ToStatus(
      wrap::hipGraphDebugDotPrint(graph_, std::string{path}.c_str(), flags),
      "Failed to print gpu graph debug file");
}

absl::Status RocmCommandBuffer::InstantiateGraph() {
  VLOG(2) << "Instantiate HIP executable graph from graph " << graph_;
  return ToStatus(
      wrap::hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0),
      "Failed to instantiate HIP graph");
}

RocmCommandBuffer::~RocmCommandBuffer() {
  if (exec_ != nullptr) {
    auto exec_num = NotifyExecDestroyed();
    VLOG(5) << "Destroy GPU command buffer executable graph " << exec_ << " "
            << "(remaining alive executable graphs: " << exec_num << ")";
    if (auto status = ToStatus(hipGraphExecDestroy(exec_),
                               "Failed to destroy HIP executable graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
  if (graph_ != nullptr && is_owned_graph_) {
    if (auto status =
            ToStatus(hipGraphDestroy(graph_), "Failed to destroy HIP graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}
absl::Status RocmCommandBuffer::CheckCanBeUpdated() {
  if (exec_ == nullptr) {
    return absl::InternalError(
        "Command buffer has to have a graph executable to be updated.");
  }
  return absl::OkStatus();
}

std::string RocmCommandBuffer::ToString() const {
  return "ROCM graph debug dot print is not supported.";
}

}  // namespace stream_executor::gpu
