#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

EXCLUDED_TESTS=(
    "HostMemoryAllocateTest.Numa" # Failing on RBE
    "*IotaR1Test*" # Taking too many CI nodes
    "Fp8s/FloatNormalizationTest.Fp8Normalization/f8e4m3fn_f8e5m2" # TODO: fix
    "Fp8s/FloatNormalizationTest.Fp8Normalization/f8e5m2_f8e5m2" # TODO: fix
    "Fp8s/FloatNormalizationTest.Fp8Normalization/f8e5m2_f8e4m3fn" # TODO: fix
    "Fp8s/FloatNormalizationTest.Fp8Normalization/f8e4m3fn_f8e4m3fn" # TODO: fix
    "TritonAndBlasSupportForDifferentTensorSizes/TritonAndBlasSupportForDifferentTensorSizes.IsDotAlgorithmSupportedByTriton/dot_bf16_bf16_f32_x6" # TODO: fix
    "TritonAndBlasSupportForDifferentTensorSizes/TritonAndBlasSupportForDifferentTensorSizes.IsDotAlgorithmSupportedByTriton/dot_bf16_bf16_f32_x9" # TODO: fix
    "RocmExecutorTest.CreateUnifiedMemoryAllocatorWorks" # TODO: fix
    # TODO: fix, unimplemented ROCm (collective_ops_e2e_test, p2p_ops_e2e_test)
    "AsyncCollectiveOps/AsyncCollectiveOps.*/*symmetric"
    "P2POps/P2POps.CollectivePermute/enable_symmetric_buffer"
    # TODO: fix, unimplemented on ROCm (ragged_all_to_all_e2e_test)
    "RaggedAllToAllTest/RaggedAllToAllTest.*/*with_multi_gpu_barrier_with_nccl"
    # TODO: fix, BFCAllocator OOM, budget too small? (collective_ops_sharded_unsharded_e2e_test)
    "CollectiveOpsTestE2EShardedUnsharded.DotBatchAndNonContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotBatchAndBatch"
    "CollectiveOpsTestE2EShardedUnsharded.DotContractingNonContractingAndContractingNonContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotContractingAndContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotNonContractingAndContracting"
    "CollectiveOpsTestE2EShardedUnsharded.DotContractingAndReplicated"
    # TODO: fix, hardcoded device count mismatch (se_gpu_pjrt_client_test)
    "StreamExecutorGpuClientTest.DistributedInit"
    "StreamExecutorGpuClientTest.MockNcclClientTest"
    "StreamExecutorGpuClientTest.GetTopologyDescriptionWithGlobalDevicesTest"
    "StreamExecutorGpuClientTest.MockNcclClientWithGpuTopologyExecuteTest"
    "StreamExecutorGpuClientTest.MockNcclClientWithGpuTopologyTest"
    # TODO: fix, unimplemented on ROCm
    "StreamExecutorGpuClientTest.GetAbiVersion"
    # TODO: fix, memory stats mismatch (se_gpu_pjrt_client_test)
    "StreamExecutorGpuClientTest.GetCompiledMemoryStatsWithTupleAndNcclUserBuffers"
)

TAG_FILTERS=$("${SCRIPT_DIR}/rocm_tag_filters.sh")

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu"
    fi
done

"${SCRIPT_DIR}/run_xla_ci_build.sh" \
    "$@" \
    --build_tag_filters="$TAG_FILTERS" \
    --test_tag_filters="$TAG_FILTERS" \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942,gfx950 \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --curses=no \
    --color=yes \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    -- \
    //xla/... \
    -//xla/backends/gpu/tests:sorting.hlo.test_mi200 # lit test can't filter with gtest filters

