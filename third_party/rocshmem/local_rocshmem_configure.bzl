"""Configuration for locally installed ROCshmem."""

def _rocshmem_impl(repository_ctx):
    """Implementation of the local_rocshmem_configure repository rule."""
    
    # Get the ROCshmem installation path from environment variable
    rocshmem_path = repository_ctx.os.environ.get("ROCSHMEM_PATH", "/opt/rocm/rocshmem")

    # Create a simple BUILD file that exposes the local installation
    build_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rocshmem_headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    strip_include_prefix = "include",
    include_prefix = "third_party",
    #deps = ["@openmpi//:mpi_lib"],
)

filegroup(
   name = "librocshmem_device",
   srcs = ["lib/librocshmem.a"],
   visibility = ["//visibility:public"],
)

cc_library(
    name = "rocshmem_config",
    hdrs = ["rocshmem_config.h"],
    include_prefix = "third_party",
)
""".format(lib_path = rocshmem_path + "/lib")
  
    repository_ctx.file("BUILD", build_content)
    
    # Create symlinks to the actual installation
    repository_ctx.symlink(rocshmem_path + "/include", "include")
    repository_ctx.symlink(rocshmem_path + "/lib", "lib")

    # Create a simple config header
    config_header = """
#ifndef THIRD_PARTY_ROCSHMEM_ROCSHMEM_CONFIG_H_
#define THIRD_PARTY_ROCSHMEM_ROCSHMEM_CONFIG_H_

constexpr static char XLA_ROCSHMEM_VERSION[] = "local";

#endif  // THIRD_PARTY_ROCSHMEM_ROCSHMEM_CONFIG_H_
"""
    repository_ctx.file("rocshmem_config.h", config_header)

local_rocshmem_configure = repository_rule(
    implementation = _rocshmem_impl,
    environ = ["ROCSHMEM_PATH"],
    local = True,
)
