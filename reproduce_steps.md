## Installation steps

- **Machine:** MI3xx
- **Image:** `ghcr.io/rocm/jax-dev-ubu24.therock-7.14:latest`
- **XLA branch:** `pemeliya/v0.6.0_command_buffers3`
---

## Steps to Reproduce

### 1. Start the container

```bash
docker run --name jax_rocm_test -w /tf -itd --init --pids-limit=-1 \
  --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --entrypoint bash \
  --ipc=host --shm-size 32G --group-add video \
  --cap-add=CAP_SYS_ADMIN \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "/tmp/packages:/pkg" -v "$HOME:/tf" \
  -v "/tmp/bazelcache:/cache" \
  ghcr.io/rocm/jax-dev-ubu24.therock-7.14:latest
```

### 2. Attach to container

```bash
docker exec -it jax_rocm_test bash
```

### 3. Install system dependencies

```bash
apt update && apt install -y wget libnuma-dev libgl-dev libglx-dev graphviz

# Install bazelisk
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.8.1/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
mv bazelisk-linux-amd64 /usr/local/bin/bazel
```

### 4. Set environment variables

```bash
ln -s /usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel /opt/rocm
```

### 5. Clone XLA

```bash
git clone --single-branch -b pemeliya/v0.6.0_command_buffers3 git@github.com:ROCm/xla.git /tf/xla
```

### 6. Build XLA

```bash
cd /tf/xla
build=1 ./yhlo_runner.sh
```

### 7. Run the benchmark

```bash
cd /tf/xla
./yhlo_runner.sh < name of HLO file >
```

Synthetic pre-partitioned example (the runner uses `--num_partitions=1 --num_replicas=8` in `yhlo_runner.sh`):
- `collectives_while.hlo`: unrolled while loop with GEMM calls, collectives and fusion kernels

Synthetic SPMD (8 partitions) workloads, run the same way (the runner uses `--num_partitions=8 --num_replicas=1`):
- `moe_while_spmd.hlo`: MoE layers in a while loop (all-to-all + all-reduce).
- `moe_ring_attn_spmd.hlo`: ring attention (manual region, inner while with `collective-permute` of K/V) + MoE.
```bash
./yhlo_runner.sh moe_ring_attn_spmd.hlo
```

### 8. Tuning / debugging / configuration parameters

There are two XLA flags which control graph execution in XLA:
Non-straight graph execution (default):
```
--xla_gpu_enable_command_buffer=cudnn,cublas,fusion,cublaslt,custom_call,collectives \
--xla_gpu_graph_enable_concurrent_region=true \
```

Straight (sequential) graph execution:
```
--xla_gpu_enable_command_buffer=cudnn,cublas,fusion,cublaslt,custom_call,collectives \
--xla_gpu_graph_enable_concurrent_region=false \
```

Execution without graph capture:
```
--xla_gpu_enable_command_buffer= \
--xla_gpu_graph_enable_concurrent_region=false \
```
These flags need to be set in `yhlo_runner.sh` script, see `XLA_FLAGS=...`
The most convenient way to use another CLR build is to declare: `HIPLIB=...` inside the script which then goes to `LD_PRELOAD`.

### 9. Checking numerical accuracy
The program also outputs the sum of the resulting tensors on each device after each repetition, e.g.:
```bash
2026-11-10 10:59:36.465358: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1099] ======== ExecuteOnDevices started (repeat = 6).
2026-11-10 10:59:36.439860: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev0 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.439927: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev1 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.439984: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev2 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.440038: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev3 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.440095: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev4 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.440147: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev5 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.440203: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev6 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
2026-11-10 10:59:36.440256: I xla/tools/multihost_hlo_runner/functional_hlo_runner.cc:1169] 3: dev7 finite: 1024/1024 sum: -0.94696875 (c12ce63180000000)
```

The input buffers are initialized with **static seed** RNG (hence they are stable).
To ensure numerical correctness, one can run the script with and without graph capture (see above) and verify that the output results agree and that they do not change between repeats.

