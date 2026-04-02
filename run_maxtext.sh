#!/bin/bash
set -x

#SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
#pushd $SCRIPT_DIR

pushd /tf/jax_and_pp/maxtext

XLA_DIR=/tf/xla
DUMP_DIR=$XLA_DIR/uu_dump
GDB=
ROCPROF=

trap sigint_handler INT
sigint_handler() {
  echo "SIGINT caught!"
  #chmod 777 $DUMP_DIR/*
}

debug=${debug:-0}
profile=${profile:-0}

if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

if [[ ${profile} -eq 1 ]]; then
    #ROCPROF="rocprofv3 -i rocprof_counters.json -d $DUMP_DIR -o out --"
    #ROCPROF="rocprofv3 --stats --kernel-trace -d $DUMP_DIR -o out --"
    # ROCPROF="rocprofv3 --stats --hip-runtime-trace --memory-copy-trace -d $DUMP_DIR -o $DUMP_DIR/output.csv --"
    #--scratch-memory-trace ??
    #ROCPROF="rocprofv3 --kernel-trace --output-format pftrace -d $DUMP_DIR --"
    #ROCPROF="rocprofv3 --stats --truncate-kernels --kernel-trace --output-format pftrace -d $DUMP_DIR --"
   # ROCPROF="rocprofv3 --kernel-trace --output-format pftrace -d $DUMP_DIR --"
    # --hip-runtime-trace
  ROCPROF="rocprofv2 --plugin perfetto --kernel-trace -d $DUMP_DIR "
    # ROCPROF="rocsys --session vv1 launch rocprofv2 --kernel-trace -d $DUMP_DIR"
fi

rm -rf $DUMP_DIR
mkdir -p $DUMP_DIR
#rm -f /dev/shm/nccl-*
#rm -rf swdev545325 swdev550718
rm -rf $XLA_DIR/profile/2025_*
rm -f graph_*_dot*

# TF_CUDNN_WORKSPACE_LIMIT_IN_MB=16384 

# HSA_FORCE_FINE_GRAIN_PCIE=1 \

export RCCL_MSCCL_ENABLE=0  #FAv3
export RCCL_MSCCL_FORCE_ENABLE=0

export HF_HOME=/home/amd-user/huggingface \
PJRT_NPROC=4 \
TF_CUDNN_WORKSPACE_LIMIT_IN_MB=1096 \
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
HSA_ENABLE_SDMA=1 \
HIP_FORCE_DEV_KERNARG=1 \
NVTE_FUSED_ATTN=1 \
NVTE_CK_EXT_ASM=1 \
NVTE_CK_ASM_ATOMIC_FP32=0 \
NVTE_CK_ASM_NO_COEX=0 \
NVTE_CK_ASM_RTZ_CVT=1 \
NVTE_CK_USES_BWD_V3=1 \
NVTE_CK_V3_RTZ_CVT=2 \
NVTE_CK_USES_BWD_V3=1 \
NVTE_CK_IS_V3_ATOMIC_FP32=0 \
NVTE_CK_HOW_V3_BF16_CVT=2 \
ATTENTION_MODE=te \
PAD=False 


# export NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2
# perfetto SQL:
# select sum(dur), sum(dur)*100.0 / (select sum(dur) from slice) as percent, name from slice group by name order by percent desc
# SELECT
#       SUM(dur) / 1e6 AS total_duration_ms,
#       COUNT(*) AS num_occurrences,
#       (SUM(dur) / COUNT(*)) / 1e6 AS avg_duration_ms
# FROM slice
# WHERE
# name LIKE '%Cijk_Alik_Bljk%'    # nvjet is for NV side

# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export MIOPEN_LOG_LEVEL=6
export MIOPEN_GEMM_ENFORCE_BACKEND=5
# export TF_ROCM_KEEP_XLA_TEMPFILES=1 

export NCCL_PROTO=Simple
#export NCCL_MIN_NCHANNELS=112 # does it help ??    
#export NCCL_MAX_NCHANNELS=8

# export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV,INIT #,COLL 
#RCCL_KERNEL_COLL_TRACE_ENABLE=1
# RCCL debug tips https://uccl-project.github.io/posts/debug-nccl/

export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MIN_VLOG_LEVEL=0 TF_CPP_MAX_LOG_LEVEL=5

#export TF_FORCE_UNIFIED_MEMORY=true
# export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_CLIENT_MEM_FRACTION=0.75
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_ENABLE_HORIZONTAL_FUSION=true
export XLA_LHS_PREFER_ASYNC_DEPTH=false

export HSA_NO_SCRATCH_RECLAIM=1
export XLA_COMMAND_BUFFERS_USE_CACHED_ALLOCS=1
export XLA_COMMAND_BUFFERS_MODULE_RE=jit_train_step.*
export XLA_COMMAND_BUFFERS_USE_RENDEZVOUS=false
export XLA_BUFFER_ASSIGN_MAX_REUSES=-1
export GPU_MAX_HW_QUEUES=4
export DEBUG_HIP_FORCE_GRAPH_QUEUES=4
export DEBUG_HIP_GRAPH_DOT_PRINT=2

#export JAX_COMPILATION_CACHE_DIR=$XLA_DIR/jax_hsaco_cache
export TF_XLA_HSACO_CACHE_DIR=/tf/hsaco_cache
export TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD=11111111111111
export HIP_FORCE_DEV_KERNARG=1

rm -f $XLA_DIR/amd_log*
# export AMD_LOG_LEVEL=4
# export AMD_LOG_LEVEL_FILE=$XLA_DIR/amd_log

# No spaces in between!!
export TF_CPP_VMODULE=latency_hiding_scheduler=0,\
reduce_scatter_combiner=0,\
command_buffer_cmd=0,gemm_algorithm_picker=0,\
command_buffer_thunk=1,\
gpu_compiler=0,command_buffer_scheduling=1,\
gpu_command_buffer=0,rocm_command_buffer=0,\
gpu_executable=0,nccl_communicator=0,collective_pipeliner=3,\
host_tracer=2,python_tracer=2

export TEST_TMPDIR=$DUMP_DIR 

        #    --xla_dump_to=$DUMP_DIR \
                  #  --xla_dump_hlo_module_re=jit_train_step.* \
         #    --xla_gpu_load_autotune_results_from=autotune_xla.txt \
                    # --xla_gpu_load_autotune_results_from=$XLA_DIR/autotune_xla_32000.txt \
#            --xla_gpu_dump_autotune_results_to=$XLA_DIR/autotune_xla.txt \
# cudnn,fusion,cublas,cublaslt,custom_call,collectives
          #  --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER,ALLTOALL,COLLECTIVEPERMUTE \
          #  --xla_dump_hlo_pass_re=.* \
#           --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_FULL_UNROLL \
#           --xla_gpu_pgle_profile_file_or_directory_path=$XLA_DIR/maxtext_dump/pgle_profile.pbtxt \
          #  --xla_gpu_pgle_profile_file_or_directory_path=$XLA_DIR/profile.pb \

# --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
#             --xla_gpu_all_reduce_combine_threshold_bytes=8589934592  \
#             --xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
          #  --xla_gpu_load_autotune_results_from=$XLA_DIR/autotune_xla.txt \
            # --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_DOUBLE_BUFFER \

# export TENSILE_SOLUTION_SELECTION_METHOD=2
# export TENSILE_STREAMK_DYNAMIC_GRID=0
# export TENSILE_STREANK_MAX_CUS=111

#PYEXEC=$(pyenv which python)
PYEXEC=$(which python)

# this seems to give best perf so far:
#            --xla_gpu_all_reduce_combine_threshold_bytes=0 \
#            --xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
#             --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
# /tf/rccl/backup/librccl_new_Lnt_Snt.so
# /tf/rccl/backup/librccl_dev_original.so
# /tf/rccl/backup/librccl_new_nokernarg.so
#           --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER,ALLTOALL,COLLECTIVEPERMUTE \
# --xla_gpu_enable_command_buffer=cudnn,cublas,fusion,cublaslt,custom_call,collectives  \

HIPLIB= #/tf/rocm-systems/projects/clr/build/hipamd/lib/libamdhip64.so #:/tf/rocr-runtime-install/lib/libhsa-runtime64.so
RCCL=/tf/rccl/backup/librccl_orig_711.so.1.0 #/tf/rccl/build/librccl.so
            # --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER,ALLTOALL,COLLECTIVEPERMUTE \

export LD_PRELOAD=/opt/rocm/lib/libMIOpen.so.1:$HIPLIB:$RCCL
export XLA_FLAGS="--xla_gpu_enable_cublaslt=true \
           --xla_gpu_enable_latency_hiding_scheduler=false \
           --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER,ALLTOALL,COLLECTIVEPERMUTE \
           --xla_gpu_autotune_level=0 \
            --xla_gpu_enable_nccl_comm_splitting=false \
            --xla_gpu_collectives_use_persistent_cliques=true \
           --xla_gpu_enable_triton_softmax_fusion=false \
           --xla_gpu_enable_pipelined_reduce_scatter=false \
            --xla_gpu_enable_pipelined_all_reduce=false \
            --xla_gpu_enable_pipelined_all_gather=false \
           --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false \
           --xla_dump_to=$DUMP_DIR  --xla_dump_hlo_module_re=jit_train_step.* \
           --xla_gpu_enable_all_gather_combine_by_dim=false \
           --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
            --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 \
            --xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
            --xla_gpu_reduce_scatter_combine_threshold_bytes=0 \
            --xla_gpu_experimental_parallel_collective_overlap_limit=1 \
           --xla_gpu_graph_min_graph_size=5 \
           --xla_gpu_enable_command_buffer= \
           --xla_gpu_graph_enable_concurrent_region=true \
            --xla_gpu_enable_triton_gemm=false \
           --xla_gpu_force_compilation_parallelism=4 \
           --xla_gpu_enable_highest_priority_async_stream=true \
           --xla_gpu_memory_limit_slop_factor=95 \
           --xla_gpu_autotune_gemm_rtol=0.01"

NumProcs=1
TotalGpus=8
pkill -9 -c -f train
rm -f $XLA_DIR/zzout_*.log

# export JAX_COORDINATOR_IP="127.0.0.1"
# export JAX_COORDINATOR_PORT=12345
export NNODES=$NumProcs
MAXTEXT_CFG=$XLA_DIR/maxtext_reduce_scatter.yml
# MAXTEXT_CFG=$XLA_DIR/xuefei_config.yml

for ((pid = 0; pid < $NumProcs; pid++ )); do

  last_id=$(($NumProcs - 1))
  div=$(($TotalGpus/$NumProcs))
  gpus=$(seq -s, $((pid*div)) $((pid*div+div-1)))
  if [[ pid -eq last_id ]]; then
    NODE_RANK=$pid \
    HIP_VISIBLE_DEVICES=$gpus \
    $GDB $ROCPROF $PYEXEC -m MaxText.train $MAXTEXT_CFG 2>&1 | tee $XLA_DIR/zzzrun.log
  else
    NODE_RANK=$pid \
    HIP_VISIBLE_DEVICES=$gpus \
    $PYEXEC -m MaxText.train $MAXTEXT_CFG 2>&1 | tee $XLA_DIR/zzout_$pid.log &
  fi
done

chmod -R 777 $DUMP_DIR
# $XLA_DIR/maxtext_swdev545325.yml
# MaxText/configs/llama3_70b_gpu_maxtext_config.yml

popd
