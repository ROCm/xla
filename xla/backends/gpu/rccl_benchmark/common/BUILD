load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Shared infrastructure for the RCCL benchmark cases.
#
# Every case must use these rather than growing its own variant. Eight
# directories with eight slightly different notions of "the payload is correct"
# would be eight different standards, and the point of the suite is that a
# result means the same thing wherever it comes from.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "data_pattern",
    testonly = True,
    srcs = ["data_pattern.cc"],
    hdrs = ["data_pattern.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "guarded_buffer",
    testonly = True,
    srcs = ["guarded_buffer.cc"],
    hdrs = ["guarded_buffer.h"],
    deps = [
        ":data_pattern",
        "//xla:types",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "path_assert",
    testonly = True,
    srcs = ["path_assert.cc"],
    hdrs = ["path_assert.h"],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "case_config",
    testonly = True,
    srcs = ["case_config.cc"],
    hdrs = ["case_config.h"],
    deps = [
        ":guarded_buffer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# Entry point that configures RCCL debug logging before the library can cache
# its parameters. Cases link this instead of gtest_main.
cc_library(
    name = "benchmark_main",
    testonly = True,
    srcs = ["benchmark_main.cc"],
    deps = [
        ":path_assert",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)
