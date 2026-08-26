# XLA-driven RCCL conformance and performance suite.
#
# The subject under test is RCCL; XLA is the driver. See README.md for what
# that distinction changes, and for how to add a mechanism.
#
# Subdirectories are named after the RCCL mechanism they exercise, not after a
# collective, because the mechanisms cut across collectives while collectives
# do not cut across mechanisms.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "run.sh",
    "extract_thresholds.py",
])
