load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "device_spec",
    hdrs = ["device_spec.h"],
    deps = [
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "emitter_loc_op_builder",
    srcs = ["emitter_loc_op_builder.cc"],
    hdrs = ["emitter_loc_op_builder.h"],
    visibility = [
        "//xla/backends:__subpackages__",
        "//xla/codegen/emitters/transforms:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform",
    ] + if_google(["@com_google_absl//absl/types:source_location"]),
)

xla_cc_test(
    name = "emitter_loc_op_builder_test",
    srcs = ["emitter_loc_op_builder_test.cc"],
    tags = ["gpu"],
    deps = [
        ":emitter_loc_op_builder",
        "//xla/backends/gpu/codegen/triton:fusion_emitter",
        "//xla/hlo/testlib:filecheck",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "kernel_emitter",
    hdrs = ["kernel_emitter.h"],
    deps = [
        ":kernel_definition",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "llvm_ir_kernel_source",
    srcs = ["llvm_ir_kernel_source.cc"],
    hdrs = ["llvm_ir_kernel_source.h"],
    deps = [
        ":kernel_source",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:JITLink",
    ],
)

cc_library(
    name = "kernel_source",
    hdrs = ["kernel_source.h"],
)

cc_library(
    name = "kernel_definition",
    hdrs = ["kernel_definition.h"],
    deps = [
        ":kernel_source",
        ":kernel_spec",
        "//xla/tsl/platform:logging",
    ],
)
