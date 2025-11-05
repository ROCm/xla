#!/bin/bash
set -x
DUMP_DIR=uu_dump
GDB=
ROCPROF=

trap sigint_handler INT
sigint_handler() {
  echo "SIGINT caught!"
  chmod 777 $DUMP_DIR/*
}

build=${build:-0}
debug=${debug:-0}
profile=${profile:-0}

if [[ ${build} -eq 1 ]]; then
  ./ybuild-test.sh //xla/tools/multihost_hlo_runner:hlo_runner_main
  exit 0
fi


if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

if [[ ${profile} -eq 1 ]]; then
   ROCPROF="rocprofv2 --plugin perfetto --kernel-trace -d $DUMP_DIR "
fi

rm -rf $DUMP_DIR
mkdir -p $DUMP_DIR

#export HIPBLASLT_LOG_LEVEL=5
#export HIPBLASLT_LOG_FILE=file
# set mask to 32 in order to show bench calls
# apt install hipblaslt-clients
#export HIPBLASLT_LOG_MASK=32
#export HIPBLASLT_LOG_FILE=hipblaslt.out

export RCCL_MSCCL_ENABLE=0  #FAv3

export HIP_FORCE_DEV_KERNARG=1
# export AMD_LOG_LEVEL=4
export TF_CPP_VMODULE=command_buffer_scheduling=1,latency_hiding_scheduler=0,\
command_buffer_cmd=1,command_buffer_thunk=1,\
gpu_compiler=0,gpu_command_buffer=0,rocm_executor=1,\
rocm_command_buffer=0,gpu_executable=0,nccl_communicator=1,gpu_cliques=0

export TEST_TMPDIR=$DUMP_DIR 

export HSA_NO_SCRATCH_RECLAIM=1
export XLA_COMMAND_BUFFERS_USE_CACHED_ALLOCS=0
#export XLA_COMMAND_BUFFERS_MODULE_RE=jit_train_step.*
export XLA_COMMAND_BUFFERS_USE_RENDEZVOUS=false
export XLA_BUFFER_ASSIGN_MAX_REUSES=-1
export XLA_THUNKS_PROFILING=0
export XLA_ENABLE_HORIZONTAL_FUSION=true
export XLA_LHS_PREFER_ASYNC_DEPTH=true
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DEBUG_HIP_FORCE_GRAPH_QUEUES=4
export DEBUG_HIP_GRAPH_DOT_PRINT=1
export TEST_SRCDIR=zzxla/service/gpu/tests/gpu_index_test_gpu_amd_any.runfiles/
export TEST_WORKSPACE=xla
export XLA_CLIENT_MEM_FRACTION=0.95

rm -f gpucore.* graph_*

#HIPLIB=/tf/clr/build/hipamd/lib/libamdhip64.so
HIPLIB=/tf/xla/libamdhip64.so.7.1.25434-a69b0b7e43

XLA_TEST_DEVICE_TYPE=ROCM \
LD_PRELOAD=/opt/rocm/lib/libMIOpen.so.1:$HIPLIB \
XLA_FLAGS="--xla_gpu_enable_cublaslt=true \
           --xla_gpu_enable_latency_hiding_scheduler=false \
           --xla_gpu_autotune_level=0 \
           --xla_gpu_enable_nccl_comm_splitting=false \
            --xla_gpu_collectives_use_persistent_cliques=true \
             --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
            --xla_gpu_reduce_scatter_combine_threshold_bytes=0 \
            --xla_gpu_all_reduce_combine_threshold_bytes=0  \
            --xla_gpu_all_gather_combine_threshold_bytes=0 \
            --xla_gpu_collective_permute_combine_threshold_bytes=0 \
            --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_FULL_UNROLL \
            --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLTOALL,ALLGATHER,COLLECTIVEPERMUTE \
            --xla_gpu_enable_all_gather_combine_by_dim=false \
           --xla_dump_to=$DUMP_DIR  \
           --xla_gpu_reduce_scatter_combine_threshold_bytes=0 \
           --xla_gpu_experimental_parallel_collective_overlap_limit=0 \
           --xla_gpu_graph_min_graph_size=2 \
           --xla_gpu_enable_command_buffer=cudnn,cublas,fusion,cublaslt,custom_call,collectives   \
           --xla_gpu_enable_triton_gemm=false \
           --xla_gpu_mock_custom_calls=true \
           --xla_gpu_graph_enable_concurrent_region=true \
           --xla_gpu_force_compilation_parallelism=16 \
           --xla_gpu_enable_highest_priority_async_stream=false \
           --xla_gpu_strict_conv_algorithm_picker=false \
           --xla_gpu_memory_limit_slop_factor=95 \
           --xla_gpu_autotune_gemm_rtol=0.01" \
$GDB $ROCPROF ./bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main --num_partitions=1 --num_replicas=4 \
    --use_spmd_partitioning=true --hlo_argument_mode=uninitialized --device_type=gpu \
    --num_repeats=15 collectives_while.hlo --gpu_client_mem_fraction=0.95 --profile_execution=true 2>&1 | tee zzzrun.log

# --xla_gpu_dump_xspace_to=$DUMP_DIR
chmod -f 777 $DUMP_DIR/*
