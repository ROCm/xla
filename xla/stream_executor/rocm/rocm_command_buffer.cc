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
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

absl::Status RocmCommandBuffer::UpdateKernelNodes(
    absl::Span<const DeviceAddressBase> old_addresses,
    absl::Span<const DeviceAddressBase> new_addresses) {
  if (old_addresses.size() != new_addresses.size()) {
    return absl::InvalidArgumentError("address arrays must have equal size");
  }

  size_t num_nodes = 0;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(graph_, nullptr, &num_nodes),
               "Failed to get graph node count"));
  if (num_nodes == 0) return absl::OkStatus();

  std::vector<hipGraphNode_t> nodes(num_nodes);
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(graph_, nodes.data(), &num_nodes),
               "Failed to get graph nodes"));

  absl::flat_hash_map<uintptr_t, size_t> old_addr_map;
  for (size_t i = 0; i < old_addresses.size(); ++i) {
    auto val = reinterpret_cast<uintptr_t>(old_addresses[i].opaque());
    if (val != 0) old_addr_map[val] = i;
  }

  for (size_t ni = 0; ni < num_nodes; ++ni) {
    hipGraphNodeType type;
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphNodeGetType(nodes[ni], &type),
                 "Failed to get node type"));
    if (type != hipGraphNodeTypeKernel) continue;

    hipKernelNodeParams kp;
    memset(&kp, 0, sizeof(kp));
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphKernelNodeGetParams(nodes[ni], &kp),
                 "Failed to get kernel node params"));

    bool modified = false;

    if (kp.extra != nullptr) {
      void* buf_ptr = nullptr;
      size_t buf_size = 0;
      void** ep = static_cast<void**>(kp.extra);
      while (*ep != HIP_LAUNCH_PARAM_END) {
        if (*ep == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
          buf_ptr = *(ep + 1);
          ep += 2;
        } else if (*ep == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
          buf_size = *static_cast<size_t*>(*(ep + 1));
          ep += 2;
        } else {
          ep++;
        }
      }

      if (buf_ptr && buf_size > 0) {
        auto* buf = static_cast<uint8_t*>(buf_ptr);
        for (size_t off = 0; off + sizeof(void*) <= buf_size;
             off += sizeof(void*)) {
          uintptr_t val;
          memcpy(&val, buf + off, sizeof(uintptr_t));
          auto it = old_addr_map.find(val);
          if (it != old_addr_map.end()) {
            uintptr_t new_val = reinterpret_cast<uintptr_t>(
                new_addresses[it->second].opaque());
            memcpy(buf + off, &new_val, sizeof(uintptr_t));
            modified = true;
          }
        }

        if (modified) {
          TF_RETURN_IF_ERROR(ToStatus(
              wrap::hipGraphKernelNodeSetParams(nodes[ni], &kp),
              "Failed to set kernel node params after patching"));
        }
      }
    } else if (kp.kernelParams != nullptr) {
      for (int a = 0; a < 64 && kp.kernelParams[a] != nullptr; ++a) {
        uintptr_t val;
        memcpy(&val, kp.kernelParams[a], sizeof(uintptr_t));
        auto it = old_addr_map.find(val);
        if (it != old_addr_map.end()) {
          uintptr_t new_val = reinterpret_cast<uintptr_t>(
              new_addresses[it->second].opaque());
          memcpy(kp.kernelParams[a], &new_val, sizeof(uintptr_t));
          modified = true;
        }
      }
      if (modified) {
        TF_RETURN_IF_ERROR(ToStatus(
            wrap::hipGraphKernelNodeSetParams(nodes[ni], &kp),
            "Failed to set kernel node params after patching"));
      }
    }
  }

  return absl::OkStatus();
}

void RocmCommandBuffer::DumpGraphKernelNodes(absl::string_view label) {
  if (!VLOG_IS_ON(3)) return;

  size_t num_nodes = 0;
  auto s = wrap::hipGraphGetNodes(graph_, nullptr, &num_nodes);
  if (s != hipSuccess || num_nodes == 0) {
    VLOG(3) << "DumpGraph[" << label << "] graph=" << graph_
            << " nodes=0 (or error)";
    return;
  }

  std::vector<hipGraphNode_t> nodes(num_nodes);
  wrap::hipGraphGetNodes(graph_, nodes.data(), &num_nodes);

  VLOG(3) << "DumpGraph[" << label << "] graph=" << graph_
          << " total_nodes=" << num_nodes;

  for (size_t i = 0; i < num_nodes; i++) {
    hipGraphNodeType type;
    wrap::hipGraphNodeGetType(nodes[i], &type);

    if (type != hipGraphNodeTypeKernel) continue;

    hipKernelNodeParams kp;
    memset(&kp, 0, sizeof(kp));
    wrap::hipGraphKernelNodeGetParams(nodes[i], &kp);

    std::string msg;
    absl::StrAppend(&msg, "  node[", i, "] func=",
                    absl::Hex(reinterpret_cast<uintptr_t>(kp.func)),
                    " grid=(", kp.gridDim.x, ",", kp.gridDim.y, ",",
                    kp.gridDim.z, ") block=(", kp.blockDim.x, ",",
                    kp.blockDim.y, ",", kp.blockDim.z, ") shmem=",
                    kp.sharedMemBytes);

    if (kp.kernelParams != nullptr) {
      absl::StrAppend(&msg, " [kernelParams] ptrs:");
      for (int a = 0; a < 16; a++) {
        if (kp.kernelParams[a] == nullptr) break;
        void* val;
        memcpy(&val, kp.kernelParams[a], sizeof(void*));
        absl::StrAppend(&msg, " ",
                        absl::Hex(reinterpret_cast<uintptr_t>(val)));
      }
    } else if (kp.extra != nullptr) {
      void* buf_ptr = nullptr;
      size_t buf_size = 0;
      void** ep = static_cast<void**>(kp.extra);
      while (*ep != HIP_LAUNCH_PARAM_END) {
        if (*ep == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
          buf_ptr = *(ep + 1);
          ep += 2;
        } else if (*ep == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
          buf_size = *static_cast<size_t*>(*(ep + 1));
          ep += 2;
        } else {
          ep++;
        }
      }
      absl::StrAppend(&msg, " [extra] buf_size=", buf_size, " ptrs:");
      if (buf_ptr && buf_size > 0) {
        auto* buf = static_cast<uint8_t*>(buf_ptr);
        for (size_t off = 0; off + sizeof(void*) <= buf_size;
             off += sizeof(void*)) {
          uintptr_t val;
          memcpy(&val, buf + off, sizeof(uintptr_t));
          if (val > 0x100000000ULL) {
            absl::StrAppend(&msg, " @", off, "=",
                            absl::Hex(val));
          }
        }
      }
    }
    VLOG(3) << msg;
  }
}

absl::StatusOr<const CommandBuffer::Command*>
RocmCommandBuffer::FlattenChildGraphNodes(
    const CommandBuffer& nested,
    absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  VLOG(1) << "FlattenChildGraphNodes: CREATE graph=" << graph_
          << " exec=" << exec_;

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
    FlatNodeInfo node_info;

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

      if (kparams.kernelParams == nullptr && kparams.extra != nullptr) {
        // Decode the HIP_LAUNCH_PARAM extra array to get the packed arg
        // buffer pointer and size, then deep-copy it for the flattened node.
        void* src_buf = nullptr;
        size_t src_size = 0;
        void** ep = static_cast<void**>(kparams.extra);
        while (*ep != HIP_LAUNCH_PARAM_END) {
          if (*ep == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
            src_buf = *(ep + 1);
            ep += 2;
          } else if (*ep == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
            src_size = *static_cast<size_t*>(*(ep + 1));
            ep += 2;
          } else {
            ep++;
          }
        }

        if (!src_buf || src_size == 0) {
          VLOG(1) << "FlattenChildGraphNodes: cannot decode extra for func="
                  << kparams.func;
          return absl::InternalError("Cannot decode extra-style kernel args");
        }

        GpuFlattenedCommand::NodeArgBuffer nab;
        nab.size = src_size;
        nab.data = std::make_unique<uint8_t[]>(src_size);
        memcpy(nab.data.get(), src_buf, src_size);

        auto owned_size = std::make_unique<size_t>(src_size);
        void* extra_arr[5] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, nab.data.get(),
            HIP_LAUNCH_PARAM_BUFFER_SIZE, owned_size.get(),
            HIP_LAUNCH_PARAM_END};
        kparams.kernelParams = nullptr;
        kparams.extra = extra_arr;

        auto status = wrap::hipGraphAddKernelNode(
            &new_node, graph_, hip_deps.data(), hip_deps.size(), &kparams);
        if (status != hipSuccess) {
          VLOG(1) << "FlattenChildGraphNodes: hipGraphAddKernelNode failed "
                     "for extra kernel func=" << kparams.func
                  << " err=" << status;
          return absl::InternalError("Cannot flatten extra-style kernel node");
        }

        node_info = {FlatNodeKind::kKernelExtra, kparams.func};
        flat_cmd.node_arg_buffers.push_back(std::move(nab));
        flat_cmd.extra_arg_sizes.push_back(std::move(owned_size));
      } else {
        TF_RETURN_IF_ERROR(ToStatus(
            wrap::hipGraphAddKernelNode(&new_node, graph_, hip_deps.data(),
                                       hip_deps.size(), &kparams),
            "Failed to add flattened kernel node"));
        node_info = {FlatNodeKind::kKernel, kparams.func};
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
      node_info = {FlatNodeKind::kMemcpy, nullptr};

    } else if (type == hipGraphNodeTypeMemset) {
      hipMemsetParams msparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemsetNodeGetParams(child_node, &msparams),
                   "Failed to get memset node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphAddMemsetNode(&new_node, graph_, hip_deps.data(),
                                     hip_deps.size(), &msparams),
          "Failed to add flattened memset node"));
      node_info = {FlatNodeKind::kMemset, nullptr};

    } else if (type == hipGraphNodeTypeEmpty) {
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphAddEmptyNode(&new_node, graph_, hip_deps.data(),
                                    hip_deps.size()),
          "Failed to add flattened empty node"));
      node_info = {FlatNodeKind::kEmpty, nullptr};
    } else {
      VLOG(1) << "FlattenChildGraphNodes: skipping node type "
              << static_cast<int>(type);
      continue;
    }

    GraphNodeHandle parent_handle = FromHipGraphHandle(new_node);
    child_to_parent[child_node] = parent_handle;
    flat_cmd.node_handles.push_back(parent_handle);
    flat_cmd.node_infos.push_back(node_info);
  }

  VLOG(1) << "FlattenChildGraphNodes: created " << flat_cmd.node_handles.size()
          << " nodes in parent graph " << graph_;
  return AppendCommand(std::move(flat_cmd));
}

absl::Status RocmCommandBuffer::UpdateFlattenedChildNodes(
    const Command* command, const CommandBuffer& nested) {
  auto* flat_cmd = dynamic_cast<const GpuFlattenedCommand*>(command);
  if (!flat_cmd) {
    VLOG(1) << "UpdateFlattenedChildNodes: command is NOT GpuFlattenedCommand"
            << ", falling back";
    return absl::InternalError("Command is not a GpuFlattenedCommand");
  }
  const auto& child_buf =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested);
  hipGraph_t child_graph = child_buf.graph_;

  VLOG(2) << "UpdateFlattenedChildNodes: exec=" << exec_
          << " flat_nodes=" << flat_cmd->node_handles.size()
          << " child_graph=" << child_graph;

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

  // Collect child kernel nodes (in topo order) for matching by func pointer.
  std::vector<std::pair<hipGraphNode_t, hipKernelNodeParams>> child_kernels;
  // Collect non-kernel nodes separately by type.
  std::vector<hipGraphNode_t> child_memcpy_nodes, child_memset_nodes;

  for (hipGraphNode_t child_node : sorted) {
    hipGraphNodeType type;
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphNodeGetType(child_node, &type),
                 "Failed to get node type"));
    if (type == hipGraphNodeTypeKernel) {
      hipKernelNodeParams kp;
      memset(&kp, 0, sizeof(kp));
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphKernelNodeGetParams(child_node, &kp),
                   "Failed to get kernel node params"));
      child_kernels.push_back({child_node, kp});
    } else if (type == hipGraphNodeTypeMemcpy) {
      child_memcpy_nodes.push_back(child_node);
    } else if (type == hipGraphNodeTypeMemset) {
      child_memset_nodes.push_back(child_node);
    }
  }

  // Match each flat node to its corresponding child by kind + func pointer.
  size_t ki = 0, mi = 0, si = 0;
  for (size_t i = 0; i < flat_cmd->node_handles.size(); ++i) {
    const auto& info = flat_cmd->node_infos[i];
    GraphNodeHandle parent_handle = flat_cmd->node_handles[i];

    if (info.kind == FlatNodeKind::kKernel ||
        info.kind == FlatNodeKind::kKernelExtra) {
      if (ki >= child_kernels.size()) {
        return absl::InternalError("Kernel count mismatch in flattened update");
      }
      auto& [child_node, kparams] = child_kernels[ki++];

      if (kparams.func != info.func) {
        LOG(WARNING) << "UpdateFlattened: func mismatch at i=" << i
                     << " expected=" << info.func << " got=" << kparams.func;
        return absl::InternalError("Kernel func mismatch in flattened update");
      }

      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphExecKernelNodeSetParams(
              exec_, ToHipGraphHandle(parent_handle), &kparams),
          "Failed to update flattened kernel node"));

    } else if (info.kind == FlatNodeKind::kMemcpy) {
      if (mi >= child_memcpy_nodes.size()) {
        return absl::InternalError("Memcpy count mismatch in flattened update");
      }
      hipMemcpy3DParms mparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemcpyNodeGetParams(
                       child_memcpy_nodes[mi++], &mparams),
                   "Failed to get memcpy node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphExecMemcpyNodeSetParams1D(
              exec_, ToHipGraphHandle(parent_handle),
              mparams.dstArray ? mparams.dstPtr.ptr : mparams.dstPtr.ptr,
              mparams.srcArray ? mparams.srcPtr.ptr : mparams.srcPtr.ptr,
              mparams.extent.width, hipMemcpyDeviceToDevice),
          "Failed to update flattened memcpy node"));

    } else if (info.kind == FlatNodeKind::kMemset) {
      if (si >= child_memset_nodes.size()) {
        return absl::InternalError("Memset count mismatch in flattened update");
      }
      hipMemsetParams msparams = {};
      TF_RETURN_IF_ERROR(
          ToStatus(wrap::hipGraphMemsetNodeGetParams(
                       child_memset_nodes[si++], &msparams),
                   "Failed to get memset node params"));
      TF_RETURN_IF_ERROR(ToStatus(
          wrap::hipGraphExecMemsetNodeSetParams(
              exec_, ToHipGraphHandle(parent_handle), &msparams),
          "Failed to update flattened memset node"));
    }
  }

  VLOG(2) << "UpdateFlattenedChildNodes: matched "
          << flat_cmd->node_handles.size() << " nodes"
          << " (k=" << ki << " m=" << mi << " s=" << si << ")";
  return absl::OkStatus();
}

absl::Status RocmCommandBuffer::BuildPatchTable(
    const Command* command,
    absl::Span<const DeviceAddressBase> known_addresses) {
  auto* flat_cmd = const_cast<GpuFlattenedCommand*>(
      dynamic_cast<const GpuFlattenedCommand*>(command));
  if (!flat_cmd) {
    return absl::InternalError("BuildPatchTable: not a GpuFlattenedCommand");
  }

  flat_cmd->patch_table.clear();

  // For each extra-style kernel node, scan its deep-copied arg buffer
  // to find byte offsets that match any of the known addresses.
  size_t extra_buf_idx = 0;
  for (size_t ni = 0; ni < flat_cmd->node_infos.size(); ++ni) {
    if (flat_cmd->node_infos[ni].kind != FlatNodeKind::kKernelExtra) continue;

    if (extra_buf_idx >= flat_cmd->node_arg_buffers.size()) {
      return absl::InternalError("BuildPatchTable: node_arg_buffers underflow");
    }

    auto& nab = flat_cmd->node_arg_buffers[extra_buf_idx];
    uint8_t* buf = nab.data.get();
    size_t buf_size = nab.size;

    // Scan for each known address as a raw pointer value.
    for (int ai = 0; ai < known_addresses.size(); ++ai) {
      void* addr = const_cast<void*>(known_addresses[ai].opaque());
      if (addr == nullptr) continue;

      // Scan at pointer-aligned offsets.
      for (size_t off = 0; off + sizeof(void*) <= buf_size;
           off += sizeof(void*)) {
        void* val;
        memcpy(&val, buf + off, sizeof(void*));
        if (val == addr) {
          ArgPatchEntry entry;
          entry.node_index = ni;
          entry.byte_offset = off;
          entry.buffer_use_index = ai;
          flat_cmd->patch_table.push_back(entry);
          VLOG(2) << "BuildPatchTable: node[" << ni << "] offset=" << off
                  << " matches address[" << ai << "]=" << addr;
        }
      }
    }
    extra_buf_idx++;
  }

  flat_cmd->has_patch_table = true;
  VLOG(1) << "BuildPatchTable: " << flat_cmd->patch_table.size()
          << " patch entries for " << extra_buf_idx << " extra-style kernels";
  return absl::OkStatus();
}

absl::Status RocmCommandBuffer::PatchFlattenedNodes(
    const Command* command,
    absl::Span<const DeviceAddressBase> new_addresses) {
  auto* flat_cmd = const_cast<GpuFlattenedCommand*>(
      dynamic_cast<const GpuFlattenedCommand*>(command));
  if (!flat_cmd || !flat_cmd->has_patch_table) {
    return absl::InternalError("PatchFlattenedNodes: no patch table");
  }

  // Build node_index → node_arg_buffers index mapping.
  absl::flat_hash_map<size_t, size_t> node_to_buf;
  {
    size_t buf_idx = 0;
    for (size_t ni = 0; ni < flat_cmd->node_infos.size(); ++ni) {
      if (flat_cmd->node_infos[ni].kind == FlatNodeKind::kKernelExtra) {
        node_to_buf[ni] = buf_idx++;
      }
    }
  }

  // Group patches by node and apply them.
  absl::flat_hash_set<size_t> updated_nodes;
  for (auto& entry : flat_cmd->patch_table) {
    size_t ni = entry.node_index;
    auto buf_it = node_to_buf.find(ni);
    if (buf_it == node_to_buf.end()) continue;

    auto& nab = flat_cmd->node_arg_buffers[buf_it->second];
    void* new_addr =
        const_cast<void*>(new_addresses[entry.buffer_use_index].opaque());
    memcpy(nab.data.get() + entry.byte_offset, &new_addr, sizeof(void*));
    updated_nodes.insert(ni);
  }

  // Push updated arg buffers to the executable graph.
  for (size_t ni : updated_nodes) {
    auto buf_it = node_to_buf.find(ni);
    auto& nab = flat_cmd->node_arg_buffers[buf_it->second];

    hipGraphNode_t hip_node = ToHipGraphHandle(flat_cmd->node_handles[ni]);
    hipKernelNodeParams kp;
    memset(&kp, 0, sizeof(kp));
    TF_RETURN_IF_ERROR(
        ToStatus(wrap::hipGraphKernelNodeGetParams(hip_node, &kp),
                 "PatchFlattenedNodes: get params failed"));

    size_t arg_size = nab.size;
    void* extra_arr[5] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, nab.data.get(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END};
    kp.kernelParams = nullptr;
    kp.extra = extra_arr;

    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipGraphExecKernelNodeSetParams(exec_, hip_node, &kp),
        "PatchFlattenedNodes: update failed"));
  }

  VLOG(1) << "PatchFlattenedNodes: patched " << updated_nodes.size()
          << " nodes with " << flat_cmd->patch_table.size() << " entries";
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
