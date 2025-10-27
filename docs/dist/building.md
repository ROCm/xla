### Building XLA with ROCM

1. Run docker

``` 
docker run -it --shm-size=2g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video --ipc=host --network host -e HF_HOME=/models \
    -v ${HOME}/work/ml101/jax-llm-examples:/jaxllm -v {ROCM_XLA_DIR}:/rocmxla \
    -v ${HOME}/work/models:/models -v {ROCM_SYSTEM_DIR}:/rocm-systems -v {ROCM_JAX_DIR}:/rocm-jax -v ${JAX_DIR}:/jax --name jaxntp ghcr.io/rocm/jax-ubu24.rocm700:nightly bash
```

2. Install ROCM 

```
cd /rocm-systems/projects
cmake -B /opt/rocm  -D CMAKE_INSTALL_PREFIX=/opt/rocm-7.0.0 -DCMAKE_BUILD_TYPE=Debug rocprofiler-sdk
```

3. build rocm wheels
```
cd /rocm-jax/jax_rocm_plugin
python3 ./build/build.py build   --wheels="jax-rocm-plugin,jax-rocm-pjrt"   --rocm_version=7   --rocm_path=/opt/rocm-7.0.0   --use_clang=True   --clang_path=/opt/rocm-7.0.0/llvm/bin/clang   --bazel_options="--override_repository=xla=/rocmxla" --local_xla_path /rocmxla
```
a. to enable debugging, try adding
    `--bazel_options=--copt=-g3 
    --bazel_options=--cxxopt=-g3`

4. build jaxlib wheels
```
cd /jax
python3 ./build/build.py build   --wheels="jaxlib"   --rocm_version=7   --rocm_path=/opt/rocm-7.0.0   --use_clang=True   --clang_path=/opt/rocm-7.0.0/llvm/bin/clang   --bazel_options="--override_repository=xla=/rocmxla" --local_xla_path /rocmxla
```

5. pip install all wheels