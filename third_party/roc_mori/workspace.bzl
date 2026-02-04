"""MORI - ROCM Modular RDMA Interface"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

#        strip_prefix = "rocSHMEM-develop",

# def repo():
#     tf_http_archive(
#         name = "rocshmem",
#         sha256 = "2146ff231d9aadd2b11f324c142582f89e3804775877735dc507b4dfd70c788b",
#         urls = tf_mirror_urls("https://github.com/ROCm/rocSHMEM/archive/refs/heads/develop.zip"),
#         build_file = "//third_party/rocshmem:rocshmem.BUILD",
#         #patch_file = ["//third_party/rocshmem:archive.patch"],
#         type = "tar",
#     )