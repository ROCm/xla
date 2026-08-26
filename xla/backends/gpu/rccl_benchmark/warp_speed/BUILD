load("//xla/tests:build_defs.bzl", "xla_test")

# WarpSpeed cases.
#
# Runtime note: these need eight GPUs and a WarpSpeed-capable RCCL, and they are
# expected to fault rather than merely mismatch on an affected build. Run one
# case per process through run.sh; do not run them under the ci_multi_gpu
# config, which pins NCCL_MAX_NCHANNELS=1, limits the job to four GPUs and
# retries failures three times. Any one of those three would hide this defect.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

xla_test(
    name = "grouped_all_gather_test",
    srcs = ["grouped_all_gather_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            # Deliberately not run by the default CI sweep: it needs a specific
            # library configuration and its own environment per case.
            "manual",
        ],
    },
    backends = ["gpu"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/rccl_benchmark/common:benchmark_main",
        "//xla/backends/gpu/rccl_benchmark/common:case_config",
        "//xla/backends/gpu/rccl_benchmark/common:data_pattern",
        "//xla/backends/gpu/rccl_benchmark/common:guarded_buffer",
        "//xla/backends/gpu/rccl_benchmark/common:path_assert",
        "//xla/backends/gpu/runtime:all_gather_thunk",
        "//xla/backends/gpu/runtime:collective_thunk",
        "//xla/backends/gpu/runtime:collective_thunk_multigpu_test_utils",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/service:buffer_assignment",
        "//xla/service:computation_placer_hdr",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:buffer_allocations",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)
