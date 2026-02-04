"""Configuration for locally installed ROC MORI."""

def _mori_impl(repository_ctx):
    """Implementation of the local_mori_configure repository rule."""
    
    # Get the MORI installation path from environment variable
    mori_path = repository_ctx.os.environ.get("ROC_MORI_PATH", "/opt/rocm/mori")

    # Create a simple BUILD file that exposes the local installation
    build_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mori_headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    strip_include_prefix = "include",
    include_prefix = "third_party",
)

# filegroup(
#    name = "libmori_device",
#    srcs = ["lib/libmori.a"],
#    visibility = ["//visibility:public"],
# )

cc_library(
    name = "roc_mori_config",
    hdrs = ["roc_mori_config.h"],
    include_prefix = "third_party",
)
""".format(lib_path = mori_path + "/lib")
  
    repository_ctx.file("BUILD", build_content)
    
    # Create symlinks to the actual installation
    repository_ctx.symlink(mori_path + "/include", "include")
    repository_ctx.symlink(mori_path + "/lib", "lib")

    # Create a simple config header
    config_header = """
#ifndef THIRD_PARTY_ROCM_MORI_MORI_CONFIG_H_
#define THIRD_PARTY_ROCM_MORI_MORI_CONFIG_H_

constexpr static char XLA_ROCM_MORI_VERSION[] = "local";

#endif  // THIRD_PARTY_ROCM_MORI_MORI_CONFIG_H_
"""
    repository_ctx.file("roc_mori_config.h", config_header)

local_mori_configure = repository_rule(
    implementation = _mori_impl,
    environ = ["ROC_MORI_PATH"],
    local = True,
)