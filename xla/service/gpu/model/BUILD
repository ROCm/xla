load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//xla:xla.bzl", "xla_cc_test")

# Libraries for performance modeling of HLO.
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "analytical_latency_estimator",
    srcs = ["analytical_latency_estimator.cc"],
    hdrs = ["analytical_latency_estimator.h"],
    deps = [
        ":gpu_collective_performance_model",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        ":gpu_performance_model_base",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "sol_latency_estimator",
    srcs = ["sol_latency_estimator.cc"],
    hdrs = ["sol_latency_estimator.h"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        ":gpu_performance_model_base",
        ":sol_gpu_cost_model",
        "//xla:util",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "sol_latency_estimator_test",
    srcs = ["sol_latency_estimator_test.cc"],
    deps = [
        ":sol_gpu_cost_model",
        ":sol_latency_estimator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "sol_gpu_cost_model",
    srcs = ["sol_gpu_cost_model.cc"],
    hdrs = ["sol_gpu_cost_model.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "sol_gpu_cost_model_test",
    srcs = ["sol_gpu_cost_model_test.cc"],
    deps = [
        ":sol_gpu_cost_model",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "analytical_latency_estimator_test",
    srcs = ["analytical_latency_estimator_test.cc"],
    backends = [
        "gpu_v100",
        "gpu_a100",
        "gpu_h100",
        "gpu_b200",
    ],
    deps = [
        ":analytical_latency_estimator",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "fusion_analysis_cache",
    srcs = ["fusion_analysis_cache.cc"],
    hdrs = ["fusion_analysis_cache.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "fusion_analysis_cache_test",
    srcs = ["fusion_analysis_cache_test.cc"],
    deps = [
        ":fusion_analysis_cache",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_cost_model_stats_collection",
    srcs = ["gpu_cost_model_stats_collection.cc"],
    hdrs = ["gpu_cost_model_stats_collection.h"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        ":gpu_performance_model_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_graph_dumper",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "gpu_cost_model_stats_collection_test",
    srcs = ["gpu_cost_model_stats_collection_test.cc"],
    deps = [
        ":gpu_cost_model_stats_collection",
        ":gpu_hlo_cost_analysis",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_hlo_cost_analysis",
    srcs = ["gpu_hlo_cost_analysis.cc"],
    hdrs = ["gpu_hlo_cost_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":hlo_op_profile_proto_cc",
        ":hlo_op_profiles",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "gpu_hlo_cost_analysis_test",
    srcs = ["gpu_hlo_cost_analysis_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":hlo_op_profiles",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_performance_model_base",
    srcs = ["gpu_performance_model_base.cc"],
    hdrs = ["gpu_performance_model_base.h"],
    deps = [
        ":fusion_analysis_cache",
        ":gpu_hlo_cost_analysis",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/backends/gpu/codegen/triton:fusion",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "gpu_performance_model_base_test",
    srcs = ["gpu_performance_model_base_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model_base",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_performance_model",
    srcs = ["gpu_performance_model.cc"],
    hdrs = ["gpu_performance_model.h"],
    deps = [
        ":coalescing_analysis",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model_base",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "gpu_performance_model_test",
    srcs = ["gpu_performance_model_test.cc"],
    deps = [
        ":fusion_analysis_cache",
        ":gpu_hlo_cost_analysis",
        ":gpu_indexing_performance_model",
        ":gpu_performance_model",
        ":gpu_performance_model_base",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_collective_performance_model",
    srcs = ["gpu_collective_performance_model.cc"],
    hdrs = ["gpu_collective_performance_model.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":coalescing_analysis",
        ":fusion_analysis_cache",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model_base",
        ":hlo_op_profiles",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:status",
    ] + if_cuda_is_configured(["@local_config_cuda//cuda:cuda_headers"]),
)

xla_cc_test(
    name = "gpu_collective_performance_model_test",
    srcs = ["gpu_collective_performance_model_test.cc"],
    deps = [
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_indexing_performance_model",
    srcs = ["gpu_indexing_performance_model.cc"],
    hdrs = ["gpu_indexing_performance_model.h"],
    deps = [
        ":coalescing_analysis",
        ":fusion_analysis_cache",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model_base",
        ":hlo_op_profiles",
        ":symbolic_tile_analysis",
        ":tiled_hlo_instruction_or_computation",
        ":triton_emitter_constraints",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/gpu/codegen/triton:fusion",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:instruction_fusion",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "gpu_indexing_performance_model_test",
    srcs = ["gpu_indexing_performance_model_test.cc"],
    deps = [
        ":fusion_analysis_cache",
        ":gpu_hlo_cost_analysis",
        ":gpu_indexing_performance_model",
        ":gpu_performance_model_base",
        ":symbolic_tile_analysis",
        ":tiled_hlo_instruction_or_computation",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "affine_map_evaluator",
    srcs = ["affine_map_evaluator.cc"],
    hdrs = ["affine_map_evaluator.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "affine_map_evaluator_test",
    srcs = ["affine_map_evaluator_test.cc"],
    deps = [
        ":affine_map_evaluator",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "constraint_expression",
    srcs = ["constraint_expression.cc"],
    hdrs = ["constraint_expression.h"],
    deps = [
        ":affine_map_evaluator",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "constraint_expression_test",
    srcs = ["constraint_expression_test.cc"],
    deps = [
        ":constraint_expression",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "size_and_stride_expression",
    srcs = ["size_and_stride_expression.cc"],
    hdrs = ["size_and_stride_expression.h"],
    deps = [
        ":constraint_expression",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "symbolic_tile",
    srcs = ["symbolic_tile.cc"],
    hdrs = ["symbolic_tile.h"],
    deps = [
        ":affine_map_evaluator",
        ":constraint_expression",
        ":size_and_stride_expression",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tile_test",
    srcs = ["symbolic_tile_test.cc"],
    deps = [
        ":affine_map_evaluator",
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "symbolic_tiled_hlo_instruction",
    srcs = ["symbolic_tiled_hlo_instruction.cc"],
    hdrs = ["symbolic_tiled_hlo_instruction.h"],
    deps = [
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "symbolic_tiled_hlo_instruction_test",
    srcs = ["symbolic_tiled_hlo_instruction_test.cc"],
    deps = [
        ":symbolic_tile",
        ":symbolic_tiled_hlo_instruction",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tiled_hlo_instruction_or_computation",
    srcs = [
        "tiled_hlo_computation.cc",
        "tiled_hlo_instruction.cc",
    ],
    hdrs = [
        "tiled_hlo_computation.h",
        "tiled_hlo_instruction.h",
    ],
    deps = [
        "//xla:util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service:name_uniquer",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/lib/gtl:iterator_range",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "tiled_hlo_instruction_test",
    srcs = ["tiled_hlo_instruction_test.cc"],
    deps = [
        ":tiled_hlo_instruction_or_computation",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "tiled_hlo_computation_test",
    srcs = ["tiled_hlo_computation_test.cc"],
    deps = [
        ":tiled_hlo_instruction_or_computation",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "symbolic_tile_analysis",
    srcs = ["symbolic_tile_analysis.cc"],
    hdrs = ["symbolic_tile_analysis.h"],
    deps = [
        ":constraint_expression",
        ":symbolic_tile",
        ":symbolic_tiled_hlo_instruction",
        ":tiled_hlo_instruction_or_computation",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/service:name_uniquer",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tile_analysis_test",
    srcs = ["symbolic_tile_analysis_test.cc"],
    deps = [
        ":symbolic_tile",
        ":symbolic_tile_analysis",
        ":symbolic_tiled_hlo_instruction",
        ":tiled_hlo_instruction_or_computation",
        "//xla:util",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "triton_emitter_constraints",
    srcs = ["triton_emitter_constraints.cc"],
    hdrs = ["triton_emitter_constraints.h"],
    deps = [
        ":affine_map_evaluator",
        ":symbolic_tile",
        ":symbolic_tile_analysis",
        ":symbolic_tiled_hlo_instruction",
        "//xla:shape_util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "triton_emitter_constraints_test",
    srcs = ["triton_emitter_constraints_test.cc"],
    deps = [
        ":symbolic_tile_analysis",
        ":triton_emitter_constraints",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "coalescing_analysis",
    srcs = ["coalescing_analysis.cc"],
    hdrs = ["coalescing_analysis.h"],
    deps = [
        ":affine_map_evaluator",
        ":tiled_hlo_instruction_or_computation",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "coalescing_analysis_test",
    srcs = ["coalescing_analysis_test.cc"],
    deps = [
        ":coalescing_analysis",
        ":symbolic_tile",
        ":symbolic_tile_analysis",
        ":tiled_hlo_instruction_or_computation",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

tf_proto_library(
    name = "hlo_op_profile_proto",
    srcs = ["hlo_op_profile.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//xla/service:hlo_proto",
    ],
)

cc_library(
    name = "hlo_op_profiles",
    srcs = ["hlo_op_profiles.cc"],
    hdrs = [
        "hlo_op_profiles.h",
        "hlo_op_profiles_data.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":hlo_op_profile_proto_cc",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "hlo_op_profiles_test",
    srcs = ["hlo_op_profiles_test.cc"],
    deps = [
        ":hlo_op_profiles",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_op_profiler_lib",
    srcs = ["hlo_op_profiler.cc"],
    hdrs = ["hlo_op_profiler.h"],
    compatible_with = get_compatible_with_portable(),
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    deps = [
        ":hlo_op_profile_proto_cc",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/profiler/gpu:cupti_collector",
        "//xla/backends/profiler/gpu:cupti_tracer",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_verifier",
        "//xla/service:interpreter_plugin",
        "//xla/stream_executor:device_description",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

[
    xla_test(
        name = "hlo_op_profiler_run_" + sm,
        timeout = "eternal",
        srcs = ["hlo_op_profiler_run.cc"],
        # Disable backend optimizations (in particular reassociate and instcombine) which would optimize
        # expressions like integer add and multiply.
        args = ["--xla_backend_optimization_level=0"],
        backends = ["gpu"],
        # This is a development tool, not a normal test, and thus should only be run
        # manually with --config=cuda.
        tags = [
            "gpu",
            "manual",
            "notap",
            "requires-gpu-" + sm + "-only",
        ],
        deps = [
            ":hlo_op_profile_proto_cc",
            ":hlo_op_profiler_lib",
            ":hlo_op_profiles",
            "//xla:debug_options_flags",
            "//xla:xla_data_proto_cc",
            "//xla/hlo/ir:hlo",
            "//xla/service:hlo_runner",
            "//xla/service:platform_util",
            "//xla/stream_executor:device_description",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/log",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@tsl//tsl/platform:env",
            "@tsl//tsl/platform:path",
            "@tsl//tsl/platform:platform_port",
            "@tsl//tsl/platform:protobuf",
            "@tsl//tsl/platform:status",
        ],
    )
    for sm in [
        "sm60",
        "sm70",
        "sm80",
        "sm90",
    ]
]

build_test(
    name = "hlo_op_profiler_build_test",
    targets = [
        ":hlo_op_profiler_run_sm80",
    ],
)

xla_test(
    name = "hlo_op_profiler_test",
    srcs = ["hlo_op_profiler_test.cc"],
    # TODO(b/372714955): Fix the memory leak!
    args = if_google(["--heap_check="]),
    backends = ["gpu"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    deps = [
        ":hlo_op_profiler_lib",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "sol_gpu_cost_model_stats_collection",
    srcs = ["sol_gpu_cost_model_stats_collection.cc"],
    hdrs = ["sol_gpu_cost_model_stats_collection.h"],
    deps = [
        ":sol_gpu_cost_model",
        ":sol_latency_estimator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_verifier",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "sol_gpu_cost_model_stats_collection_test",
    srcs = ["sol_gpu_cost_model_stats_collection_test.cc"],
    deps = [
        ":sol_gpu_cost_model_stats_collection",
        "//xla:shape_util",
        "//xla/hlo/testlib:filecheck",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "interpolator",
    hdrs = ["interpolator.h"],
    deps = ["@com_google_absl//absl/log:check"],
)

xla_cc_test(
    name = "interpolator_test",
    srcs = ["interpolator_test.cc"],
    deps = [
        ":interpolator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_interpolator",
    srcs = ["collective_interpolator.cc"],
    hdrs = ["collective_interpolator.h"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":hlo_op_profile_proto_cc",
        ":interpolator",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu/transforms/collectives:collective_ops_utils",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "collective_interpolator_test",
    srcs = ["collective_interpolator_test.cc"],
    deps = [
        ":collective_interpolator",
        ":hlo_op_profile_proto_cc",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/transforms/collectives:collective_ops_utils",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_ptable_stats_collection",
    srcs = ["collective_ptable_stats_collection.cc"],
    hdrs = ["collective_ptable_stats_collection.h"],
    deps = [
        ":collective_interpolator",
        ":hlo_op_profile_proto_cc",
        ":hlo_op_profiles",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "collective_ptable_stats_collection_test",
    srcs = ["collective_ptable_stats_collection_test.cc"],
    deps = [
        ":collective_ptable_stats_collection",
        ":hlo_op_profile_proto_cc",
        ":hlo_op_profiles",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)
