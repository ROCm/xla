# Build file for libsvm - automatically downloaded from GitHub
# Repository: https://github.com/cjlin1/libsvm

package(default_visibility = ["//visibility:public"])

# Compile libsvm from source
cc_library(
    name = "libsvm",
    srcs = ["svm.cpp"],
    hdrs = ["svm.h"],
    includes = ["."],
    copts = [
        "-O3",
        "-fPIC",
        "-Wall",
    ],
    visibility = ["//visibility:public"],
)

