load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

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

td_library(
    name = "triton_xla_td_files",
    srcs = glob(["*.td"]),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "triton_xla_dialect_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "triton_xla_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "triton_xla_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "triton_xla_dialect.td",
    deps = [":triton_xla_td_files"],
)

gentbl_cc_library(
    name = "triton_xla_ops_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "triton_xla_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "triton_xla_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "triton_xla_ops.td",
    deps = [
        ":triton_xla_td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@triton//:td_files",
    ],
)

gentbl_cc_library(
    name = "triton_xla_types_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=triton_xla",
            ],
            "triton_xla_types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=triton_xla",
            ],
            "triton_xla_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "triton_xla_types.td",
    deps = [":triton_xla_td_files"],
)

gentbl_cc_library(
    name = "triton_xla_attrs_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=triton_xla",
            ],
            "triton_xla_attrs.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect=triton_xla",
            ],
            "triton_xla_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "triton_xla_attrs.td",
    deps = [
        ":triton_xla_td_files",
        "@triton//:td_files",
    ],
)

cc_library(
    name = "triton_xla",
    srcs = [
        "triton_xla_attrs.cc",
        "triton_xla_dialect.cc",
        "triton_xla_ops.cc",
        "triton_xla_types.cc",
    ],
    hdrs = ["triton_xla_ops.h"],
    deps = [
        ":triton_xla_attrs_inc_gen",
        ":triton_xla_dialect_inc_gen",
        ":triton_xla_ops_inc_gen",
        ":triton_xla_types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
    ],
)
