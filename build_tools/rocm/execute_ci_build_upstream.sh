#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

EXCLUDED_TESTS=(
    "HostMemoryAllocateTest.Numa"                                                                                                                  # Failing on RBE
    "NumericTestsForBlas/NumericTestsForBlas.Infinity/dot_tf32_tf32_f32_x3"
    "TritonEmitterTest.ScaledDotIsSupportedByReferencePlatform"
    "VmmTest.CommandBufferSkipProfiledTwoGemmChain"
    "GpuCollectivesTest.CreateSymmetricMemory"
    "GpuCollectivesTest.CreateSymmetricMemoryOnDifferentComms"
    "GpuCollectivesTest.CreateRegisteredMemory"
    "GpuCollectivesTest.CreateWithMultipleIds"
    "GpuCollectivesTest.PutAndWaitSignal"
    "GpuCollectivesTest.SplitCommunicators"
    "RocmMemoryReservationTest.RemapRepointsRequiredSlice"
    "ConvolutionTest.Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid"
    "GpuKernelTilingTest.ReductionInputTooLarge"
    "MxScaledDotExecutionTest.MxFp4Fp8MixedBatchedCorrectness"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllReduce/async_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllReduce/sync_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllMemCpyWithoutSplitDim/async_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllMemCpyWithoutSplitDim/sync_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllNumberOfElementsLargerThanInt32Max/async_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllNumberOfElementsLargerThanInt32Max/sync_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllWithoutSplitDim/async_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllWithoutSplitDim/sync_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllWithSplitDim/async_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncAllToAllWithSplitDim/sync_symmetric"
    "CollectiveOpsTestE2EShardedUnsharded.DotBatchAndBatch"
    "CollectiveOpsTestE2EShardedUnsharded.DotBatchAndNonContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotContractingAndContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotContractingAndReplicated"
    "CollectiveOpsTestE2EShardedUnsharded.DotContractingNonContractingAndContractingNonContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotNonContractingAndContracting"
    "CollectivePipelineParallelismTestWithAndWithoutOpts/CollectivePipelineParallelismTest.PartiallyPipelinedAsyncSendRecvLoop/1"
    "P2POps/P2POps.CollectivePermute/enable_symmetric_buffer"
    "RcclSymmetricMemoryTest.ToStringContainsExpectedFields"
    "RcclSymmetricMemoryTest.PackKernelArgReturnsValidWindowHandle"
    "RcclSymmetricMemoryTest.MultimemAddrNotSupported"
    "RcclSymmetricMemoryTest.TwoWindowsHaveDistinctHandles"
    "RcclSymmetricMemoryTest.CreateSucceeds"
    "RcclSymmetricMemoryTest.AddrMatchesRegisteredBuffer"
)

TAG_FILTERS=$("${SCRIPT_DIR}/rocm_tag_filters.sh")

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,-multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_rocm_cpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-requires-gpu-rocm,-requires-gpu-amd"
    fi
done

"${SCRIPT_DIR}/run_xla_ci_build.sh" \
    "$@" \
    --build_tag_filters="$TAG_FILTERS" \
    --test_tag_filters="$TAG_FILTERS" \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx950 \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --test_timeout=920,2400,7200,9600 \
    --sandbox_add_mount_pair=/dev/null:/etc/ld.so.cache \
    --curses=no \
    --color=yes \
    --jobs=30 \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    --cache_test_results=yes \
    --nokeep_going \
    --repo_env=TF_ROCM_RBE_SINGLE_GPU_POOL=linux_x64_gpu_do_gfx950 \
    --repo_env=ROCM_PATH= \
    --repo_env=ROCM_DISTRO_URL="https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx950-dcgpu-7.14.0.tar.gz" \
    --repo_env=ROCM_DISTRO_HASH="12afeccd06e6caf0699d86d688f16083aafa35474d0ec1d8063477fb5c119d49" \
    -- \
    //xla/...
