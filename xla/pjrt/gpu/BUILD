load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/pjrt/gpu:package_groups.bzl", "xla_gpu_internal_packages")
load("//xla/stream_executor:build_defs.bzl", "if_cuda_or_rocm")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# Integrate with PJRT rather than the GPU client directly.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],
)

xla_gpu_internal_packages()

cc_library(
    name = "gpu_helpers",
    srcs = ["gpu_helpers.cc"],
    hdrs = ["gpu_helpers.h"],
    visibility = internal_visibility(["//xla/pjrt:friends"]),
    deps = [
        "//xla:types",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/integrations:device_mem_allocator",
        "//xla/stream_executor/integrations:stream_executor_allocator",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework:bfc_allocator",
        "//xla/tsl/framework:device_id_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "se_gpu_pjrt_client",
    srcs = ["se_gpu_pjrt_client.cc"],
    hdrs = ["se_gpu_pjrt_client.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]),
    visibility = internal_visibility(["//xla/pjrt/gpu:legacy_gpu_client_users"]),
    deps = [
        ":gpu_helpers",
        ":gpu_metrics",
        ":gpu_topology",
        ":gpu_topology_proto_cc",
        ":se_gpu_topology_description",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/pjrt:event_pool",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:local_device_state",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt:stream_executor_executable",
        "//xla/pjrt:stream_executor_executable_proto_cc",
        "//xla/pjrt:tracked_device_buffer",
        "//xla/pjrt:utils",
        "//xla/pjrt:worker_thread",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:in_memory_key_value_store",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:topology_util",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:buffer_assignment",
        "//xla/service:compiler",
        "//xla/service:computation_placer_hdr",
        "//xla/service:executable",
        "//xla/service:global_device_id",
        "//xla/service:platform_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/integrations:device_mem_allocator",
        "//xla/stream_executor/integrations:tf_allocator_adapter",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework:bfc_allocator",
        "//xla/tsl/framework:device_id",
        "//xla/tsl/framework:device_id_impl",
        "//xla/tsl/lib/strings:proto_serialization",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:nvtx_utils",
        "@tsl//tsl/profiler/lib:traceme",
    ] + if_cuda_or_rocm([
        # keep sorted
        ":nccl_id_store",
        "//xla:debug_options_flags",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:gpu_memory_space_assignment",
        "//xla/service/gpu:stream_executor_util",
    ]) + if_cuda([
        # keep sorted
        "//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm([
        # keep sorted
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_cc_test(
    name = "se_gpu_pjrt_client_test",
    srcs = ["se_gpu_pjrt_client_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":gpu_topology",
        ":gpu_topology_proto_cc",
        ":se_gpu_pjrt_client",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:in_memory_key_value_store",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/profiling:device_time_measurement",
        "//xla/pjrt/profiling/test_util:mock_device_time_measurement",
        "//xla/service:gpu_plugin",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:literal_test_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_id_store",
    srcs = ["nccl_id_store.cc"],
    hdrs = ["nccl_id_store.h"],
    deps = [
        "//xla:status_macros",
        "//xla:util",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "pjrt_client_test_se_gpu",
    srcs = ["pjrt_client_test_se_gpu.cc"],
    backend_tags = {
        "gpu": ["multi_gpu"],
    },
    backends = ["gpu"],
    tags = [
        "no_oss",
        "notap",
    ],
    deps = [
        ":se_gpu_pjrt_client",
        "//xla/pjrt:pjrt_client_test_common",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "gpu_topology_proto",
    srcs = ["gpu_topology.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_topology",
    srcs = ["gpu_topology.cc"],
    hdrs = ["gpu_topology.h"],
    visibility = internal_visibility([
        "//xla/pjrt/gpu:legacy_gpu_topology_users",
        ":__subpackages__",
    ]),
    deps = [
        ":gpu_topology_proto_cc",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "se_gpu_pjrt_compiler_impl",
    srcs = ["se_gpu_pjrt_compiler.cc"],
    hdrs = ["se_gpu_pjrt_compiler.h"],
    deps = [
        ":se_gpu_pjrt_client",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:stream_executor_executable",
        "//xla/pjrt:utils",
        "//xla/service:compiler",
        "//xla/service:dump",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:local_service_utils",
        "//xla/service:platform_util",
        "//xla/service/gpu:executable_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/stream_executor/sycl:sycl_platform_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "se_gpu_pjrt_compiler_cuda_registration",
    srcs = ["se_gpu_pjrt_compiler_cuda_registration.cc"],
    compatible_with = [],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":se_gpu_pjrt_compiler_impl",
        "//xla/pjrt:pjrt_compiler",
        "//xla/service/gpu:nvptx_compiler",  # buildcleaner: keep
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/platform:initialize",
    ],
    alwayslink = 1,
)

cc_library(
    name = "se_gpu_pjrt_compiler_rocm_registration",
    srcs = ["se_gpu_pjrt_compiler_rocm_registration.cc"],
    compatible_with = [],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":se_gpu_pjrt_compiler_impl",
        "//xla/pjrt:pjrt_compiler",
        "//xla/service/gpu:amdgpu_compiler",  # buildcleaner: keep
        "//xla/stream_executor/platform:initialize",
        "//xla/stream_executor/rocm:rocm_platform_id",
    ],
    alwayslink = 1,
)

cc_library(
    name = "se_gpu_pjrt_compiler",
    hdrs = ["se_gpu_pjrt_compiler.h"],
    # TODO(b/382070262): Migrate users off this target - They should go though the PjRt ABI compatibility layer.
    visibility = internal_visibility(["//xla/pjrt/gpu:legacy_se_gpu_pjrt_compiler_users"]),
    deps = [
        ":se_gpu_pjrt_compiler_impl",
        "//xla/hlo/builder:xla_computation",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/service:compiler",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
    ] + if_cuda([
        ":se_gpu_pjrt_compiler_cuda_registration",
    ]) + if_rocm([
        ":se_gpu_pjrt_compiler_rocm_registration",
    ]),
)

cc_library(
    name = "gpu_metrics",
    srcs = ["gpu_metrics.cc"],
    hdrs = ["gpu_metrics.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/lib/monitoring:gauge",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "se_gpu_pjrt_compiler_test",
    srcs = ["se_gpu_pjrt_compiler_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_topology",
        ":se_gpu_pjrt_client",
        ":se_gpu_pjrt_compiler_impl",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test",
        "//xla/mlir_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:platform_util",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "se_gpu_pjrt_compiler_aot_test",
    srcs = ["se_gpu_pjrt_compiler_aot_test.cc"],
    backends = ["gpu"],
    deps = [
        ":se_gpu_pjrt_client",
        ":se_gpu_pjrt_compiler_impl",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/mlir_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:compiler",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "se_gpu_topology_description",
    srcs = ["se_gpu_topology_description.cc"],
    hdrs = ["se_gpu_topology_description.h"],
    visibility = internal_visibility(["//xla/pjrt/gpu:legacy_gpu_topology_users"]),
    deps = [
        ":gpu_topology",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_stream_executor_device_description",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)
