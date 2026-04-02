#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
pushd $SCRIPT_DIR

# set -x
DUMP_DIR=uu_dump
GDB=
PROF=

trap sigint_handler INT
sigint_handler() {
  echo "SIGINT caught!"
  chmod 777 $DUMP_DIR/*
}

build=${build:-0}
debug=${debug:-0}
profile=${profile:-0}

if [[ ${build} -eq 1 ]]; then
  ./xxbuild-test.sh //xla/tools/multihost_hlo_runner:hlo_runner_main
  exit 0
fi


if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

if [[ ${profile} -eq 1 ]]; then
    #PROF="rocprofv3 -i rocprof_counters.json -d $DUMP_DIR -o out --"
    #PROF="rocprofv3 --stats --kernel-trace -d $DUMP_DIR -o out --"
    # PROF="rocprofv3 --stats --hip-runtime-trace --memory-copy-trace -d $DUMP_DIR -o $DUMP_DIR/output.csv --"
    #--scratch-memory-trace ??
    #PROF="rocprofv3 --kernel-trace --output-format pftrace -d $DUMP_DIR --"
    #PROF="rocprofv3 --stats --truncate-kernels --kernel-trace --output-format pftrace -d $DUMP_DIR --"
   #PROF="rocprofv3 --kernel-trace --output-format pftrace -d $DUMP_DIR --"
    # --hip-runtime-trace
   PROF="rocprofv2 --plugin perfetto --kernel-trace -d $DUMP_DIR "
    # PROF="rocsys --session vv1 launch rocprofv2 --kernel-trace -d $DUMP_DIR"
fi

rm -rf $DUMP_DIR
mkdir -p $DUMP_DIR

export RCCL_MSCCL_ENABLE=0  #FAv3
export HF_HOME=/home/amd-user/huggingface \
PJRT_NPROC=16 \
TF_CUDNN_WORKSPACE_LIMIT_IN_MB=16384 \
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
HSA_FORCE_FINE_GRAIN_PCIE=1 \
NCCL_IB_TC=41 \
NCCL_IB_SL=0 \
GPU_MAX_HW_QUEUES=4 \
HSA_ENABLE_SDMA=1 \
HIP_FORCE_DEV_KERNARG=1 \
NVTE_FUSED_ATTN=1 \
NVTE_CK_EXT_ASM=1 \
NVTE_CK_ASM_ATOMIC_FP32=0 \
NVTE_CK_ASM_NO_COEX=0 \
NVTE_CK_ASM_RTZ_CVT=1 \
NVTE_CK_BWD_V3=1 \
NVTE_CK_V3_RTZ_CVT=2 \
XLA_CLIENT_MEM_FRACTION=0.8 \
ATTENTION_MODE=te \
PAD=False 

# MI300 tuning guide
# https://amd.atlassian.net/wiki/spaces/~hongxyan/pages/266764303/AMD+Instinct+MI300X+tuning+guide

# export TF_XLA_HSACO_CACHE_DIR=
# export TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD=7
export HIP_FORCE_DEV_KERNARG=1
# export AMD_LOG_LEVEL=4
# export AMD_LOG_LEVEL_FILE=zzamdlogs.txt
# export DEBUG_CLR_SYSMEM_POOL=true 

export TF_CPP_VMODULE=command_buffer_scheduling=1,latency_hiding_scheduler=0,\
command_buffer_cmd=1,command_buffer_thunk=1,gemm_algorithm_picker=0,\
gpu_compiler=0,gpu_command_buffer=0,rocm_executor=0,gpu_transfer_manager=0,\
rocm_command_buffer=0,gpu_executable=0,nccl_communicator=0,gpu_cliques=0,gpublas_lt_matmul_thunk=0


#export HIPBLASLT_LOG_LEVEL=5
#export HIPBLASLT_LOG_FILE=file
# export TENSILE_SOLUTION_SELECTION_METHOD=2
# export TENSILE_STREAMK_DYNAMIC_GRID=0
# export TENSILE_STREANK_MAX_CUS=111
# export HIPBLASLT_LOG_MASK=32
# export TENSILE_DB=255

#export NCCL_PROTO=LL
#export NCCL_MIN_NCHANNELS=112 # does it help ??

# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export MIOPEN_LOG_LEVEL=6
# export MIOPEN_FIND_ENFORCE=3
# export MIOPEN_USER_DB_PATH="/tf/miopen_cache"
# export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_USER_DB_PATH"
# export MIOPEN_GEMM_ENFORCE_BACKEND=5
# export TF_ROCM_KEEP_XLA_TEMPFILES=1 

# unset MIOPEN_FIND_MODE
# unset MIOPEN_FIND_ENFORCE
# unset MIOPEN_SYSTEM_DB_PATH

export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=PROFILE 
export NCCL_REPORT_CONNECT_PROGRESS=1
#RCCL_KERNEL_COLL_TRACE_ENABLE=1

#export HIPBLASLT_LOG_LEVEL=5
#export ROCBLAS_LAYER=7
#export ROCBLAS_STREAM_ORDER_ALLOC=1
export TEST_TMPDIR=$DUMP_DIR 
        #    --xla_gpu_dump_autotune_results_to=autotune_xla.txt \
        #    --xla_gpu_load_autotune_results_from=autotune_xla.txt \
#            --xla_gpu_dot_merger_threshold_mb=0 \
        #    --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
        #    --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
        #    --xla_gpu_all_reduce_combine_threshold_bytes=8589934592  \
        #    --xla_gpu_all_gather_combine_threshold_bytes=137438953472 \
        #    --xla_gpu_enable_all_gather_combine_by_dim=false \
        #    --xla_gpu_pgle_profile_file_or_directory_path=/data/xla/pgle_profile_base.pbtxt \

# RCCL without this flag is slow !!!
export HSA_NO_SCRATCH_RECLAIM=1
export XLA_COMMAND_BUFFERS_USE_CACHED_ALLOCS=1
#export XLA_COMMAND_BUFFERS_MODULE_RE=jit_train_step.*
export XLA_COMMAND_BUFFERS_USE_RENDEZVOUS=false
export XLA_BUFFER_ASSIGN_MAX_REUSES=-1
export HIP_VISIBLE_DEVICES=2,3,4,5

# export AMD_DIRECT_DISPATCH=false
# export DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2
# export AMD_SERIALIZE_KERNEL=3
export DEBUG_HIP_FORCE_GRAPH_QUEUES=4
export DEBUG_HIP_GRAPH_DOT_PRINT=2
export TEST_SRCDIR=zzxla/service/gpu/tests/gpu_index_test_gpu_amd_any.runfiles/
export TEST_WORKSPACE=xla
export XLA_CLIENT_MEM_FRACTION=0.5

  #            --xla_dump_hlo_pass_re=.* 
rm -f gpucore.* graph_*

# /opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so:/opt/rocm/lib/librocprofiler-sdk.so:
# try this out instead of handling collectives in scheduler
  #    --xla_gpu_enable_command_buffer=cudnn,cublas,fusion,cublaslt,custom_call,collectives 
          #  --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_DOUBLE_BUFFER \
         # --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_FULL_UNROLL \
         #  --xla_gpu_experimental_parallel_collective_overlap_limit=2 \
            # --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLTOALL,ALLGATHER,COLLECTIVEPERMUTE \

HIPLIB=/tf/rocm-systems/projects/clr/build/hipamd/lib/libamdhip64.so #:/tf/rocr-runtime-install/lib/libhsa-runtime64.so
RCCL=/tf/rccl/backup/librccl_orig_711.so.1.0 #/tf/rccl/build/librccl.so
# export DEBUG_HIP_DYNAMIC_QUEUES=2
# export DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0

# export TF_ROCM_NAN_CHECK=1 \
# TF_ROCM_NAN_CHECK_USE_SIMPLE_GEMMS=1 \
# TF_ROCM_NAN_CHECK_COUNT=20 \
# TF_ROCM_NAN_CHECK_VERBOSE=1 \
# TF_ROCM_NAN_CHECK_MAG_THRESHOLD=1e10 
# PROF="bpftrace -e 'uprobe:$HIPLIB:hipGraphLaunch {  @[ustack] = count(); }' -c "

# -g --  -F 999
# PROF="perf record --call-graph fp,16384 -g "

XLA_TEST_DEVICE_TYPE=ROCM \
LD_PRELOAD=/opt/rocm/lib/libMIOpen.so.1:$RCCL:$HIPLIB \
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
           --xla_gpu_enable_command_buffer= \
           --xla_gpu_enable_triton_gemm=false \
           --xla_gpu_mock_custom_calls=true \
           --xla_gpu_graph_enable_concurrent_region=true \
           --xla_gpu_force_compilation_parallelism=16 \
           --xla_gpu_enable_highest_priority_async_stream=false \
           --xla_gpu_strict_conv_algorithm_picker=false \
           --xla_gpu_memory_limit_slop_factor=95 \
           --xla_gpu_autotune_gemm_rtol=0.01" \
$GDB $PROF zzxla/tools/multihost_hlo_runner/hlo_runner_main --num_partitions=1 --num_replicas=4 \
    --use_spmd_partitioning=true --hlo_argument_mode=uninitialized --device_type=gpu \
    --xla_gpu_dump_xspace_to=$DUMP_DIR --num_repeats=30 input.hlo --gpu_client_mem_fraction=0.95 --profile_execution=false 2>&1 | tee zzzrun.log 

# 
chmod -f 777 $DUMP_DIR/*

# the following code snipped needs to be added to start profiling at some execution point:
# system(absl::StrCat("./fff_profile.sh ", getpid(), " &").c_str());
# std::this_thread::sleep_for(std::chrono::seconds(2));

# MY_PID=$!
# CMD="perf record --call-graph fp,16384 -g --pid $MY_PID -- sleep 15"
# echo "Start with:  $CMD"


# LIB=$(realpath $HIPLIB)
# echo "------------ found lib: ", $LIB 

# bpftrace -e "
# uprobe:$LIB:hipGraphLaunch
# {
#     printf(\"hipGraphLaunch hit!\\n\");
#     @[ustack] = count();
# }
# "

# Flags:
# 	--input_format="text"            	string	HLO input mode: text, proto_text, proto_binary, snapshot_proto_binary, unoptimized_snapshot_proto_binary, or unoptimized_snapshot_proto_text
# 	--run=true                       	bool	Should we run the compiled HLO?
# 	--dump_output_literal_to=""      	string	A path to which the HLO output will be dumped. Example: /a/b/literal.txt.
# 	--task_id=0                      	int32	Borg task id.
# 	--device_type="gpu"              	string	Device type: gpu, host
# 	--num_nodes=1                    	int32	Number of nodes (hosts). If greater than 1, a distributed service will be created for task_id 0
# 	--enable_mock_nccl=false         	bool	Should we simulate multi-hosts run with mock nccl collectives?
# 	--address=""                     	string	Coordinator address with port for when num_nodes > 1. Example: 127.0.0.1:12345
# 	--num_replicas=-1                	int32	The number of replicas; set to -1 for multihost execution, which then uses all devices on all host.
# 	--num_partitions=1               	int32	Number of partitions for SPMD.
# 	--log_output=false               	bool	Log the input and output to stderr.
# 	--run_xla_backend_only=false     	bool	Call only XLA's RunBackend during the compilation. This is used to run a post-optimization HLO module (dumped as 'xxx.after_optimizations.hlo.xxx'
# 	--disable_all_hlo_passes=false   	bool	Disable HLO passes or not.
# 	--use_spmd_partitioning=false    	bool	Partition the module using SPMD.
# 	--is_spmd_partitioned_module=false	bool	The module is the partitioned result of SPMD. Setting this flag also disables all HLO passes and sets use_spmd_partitioning.
# 	--xla_dump_to=""                 	string	A directory to dump xla debug data to.
# 	--xla_dump_as_text=false         	bool	Whether to dump xla debug data as text.
# 	--xla_dump_as_proto=false        	bool	Whether to dump xla debug data as protobuf.
# 	--hlo_argument_mode="use_random_inputs"	string	Specify how arguments to the HLO module are generated. Accepted values: use_device_id_as_input, use_random_inputs, use_shared_random_inputs, use_zeros_as_input or uninitialized.
# 	--while_execution_count=-1       	int32	If set to a positive number, flatten all while loops to a certain number of iterations.
# 	--remove_infeed_outfeed=true     	bool	If set, we will remove all infeed and outfeed operations.
# 	--compile_as_stablehlo=false     	bool	If set, convert the module to StableHLO before passing to PjRt for compilation.
# 	--use_layouts_from_hlo_module=false	bool	If set, use layouts from the HLO module's entry_computation_layout.
# 	--num_repeats=1                  	int32	Repeatedly execute the HLO for this many times.
# 	--execution_options_path=""      	string	A path to a protobuf text file which stores the ExecutionOptions message for this HLO module.
# 	--gpu_client_initialization_timeout_sec=300	int64	A timeout, in seconds, for the GPU client initialization. Only used for multi-node GPU runs
# 	--gpu_client_mem_fraction=0.750000	float	The maximum fraction of available memory to allocate in range of (0.0, 1.0). Same as XLA_CLIENT_MEM_FRACTION in the Python client. Only used with the BFC allocator.
# 	--profile_execution=false        	bool	If set, we will profile the execution and print the results.
# 	--xla_gpu_dump_xspace_to=""      	string	A directory to dump xspace data for GPU profiling.

popd