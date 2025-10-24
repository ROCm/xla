load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")
#load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

platform(
    name = "linux_x64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        # rocm/tensorflow-build:2.18-jammy-python3.9-rocm7.0.2
        "container-image": "docker://rocm/tensorflow-build@sha256:e4024d20eb9cfbf33dba739d5ecb962f6d32b6d768e3d51556386b3e50a780ec",
        "OSFamily": "Linux",
    },
)

platform(
    name = "linux_x64_gpu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        # rocm/tensorflow-build:2.18-jammy-python3.9-rocm7.0.2
        "container-image": "docker://rocm/tensorflow-build@sha256:e4024d20eb9cfbf33dba739d5ecb962f6d32b6d768e3d51556386b3e50a780ec",
        "OSFamily": "Linux",
        "Pool": "linux_x64_gpu",
    },
)
