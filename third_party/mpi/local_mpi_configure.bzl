"""Configuration for locally installed ROCshmem."""

def _mpi_impl(repository_ctx):
    """Implementation of the local_mpi_configure repository rule."""
    # Find MPI include directory
    mpi_path = repository_ctx.os.environ.get("MPI_PATH", "/usr/lib/x86_64-linux-gnu/openmpi")
    build_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mpi_lib",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    linkopts = [
        "-L{lib_path}",
        "-lmpi",
    ],
)
""".format(lib_path = mpi_path + "/lib")

    repository_ctx.file("BUILD", build_content)
    repository_ctx.symlink(mpi_path + "/include", "include")
    
local_mpi_configure = repository_rule(
    implementation = _mpi_impl,
    environ = ["MPI_PATH"],
    local = True,
)
