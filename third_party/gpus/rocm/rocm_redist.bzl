rocm_redist = {
    "rocm_7.10.0_gfx90X": [
        struct(
            url = "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx90X-dcgpu-7.10.0a20251106.tar.gz",
            sha256 = "a9270cac210e02f60a7f180e6a4d2264436cdcce61167440e6e16effb729a8ea",
            sub_package = None,
        ),
    ],
    "rocm_7.10.0_gfx94X": [
        struct(
            url = "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx94X-dcgpu-7.10.0a20251107.tar.gz",
            sha256 = "486dbf647bcf9b78f21d7477f43addc7b2075b1a322a119045db9cdc5eb98380",
            sub_package = None,
        ),
    ],
    "rocm_7.10.0_gfx90X_whl": [
        # Order matters
        struct(
            url = "https://rocm.nightlies.amd.com/v2/gfx90X-dcgpu/rocm_sdk_devel-7.10.0a20251018-py3-none-linux_x86_64.whl",
            sha256 = "62066278a26f5e7e00a68d774f3bd18b81263024f9778112030b13b5d907a319",
            strip_prefix = "rocm_sdk_devel",
            sub_package = struct(
                path = "_devel.tar",
                strip_prefix = "_rocm_sdk_devel",
            ),
        ),
        struct(
            url = "https://rocm.nightlies.amd.com/v2/gfx90X-dcgpu/rocm_sdk_core-7.10.0a20251018-py3-none-linux_x86_64.whl",
            sha256 = "1883929a6fa5c8bf4e5f4de1987c8d41e8f73d30a1f0aad76bdc6a2e98504bdd",
            strip_prefix = "_rocm_sdk_core",
            sub_package = None,
        ),
        struct(
            url = "https://rocm.nightlies.amd.com/v2/gfx90X-dcgpu/rocm_sdk_libraries_gfx90x_dcgpu-7.10.0a20251018-py3-none-linux_x86_64.whl",
            sha256 = "33a77b4860c280f46a29897a35812aef2bae36747dc5208408316ad51c4387c7",
            strip_prefix = "_rocm_sdk_libraries_gfx90X_dcgpu",
            sub_package = None,
        ),
    ],
}
