/* Copyright 2025 The OpenXLA Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/collectives/rocshmem_kernels.h"

namespace rocshmem {
namespace {

// rocshmem call are generated here
// https://github.com/ROCm/rocSHMEM/blob/e1a7e20b1b4372d38df4bc34872d4edd19b9c7e2/src/rocshmem.cpp#L1289
// device-side reduction: IPCContext::reduce
// https://github.com/ROCm/rocSHMEM/blob/e1a7e20b1b4372d38df4bc34872d4edd19b9c7e2/src/ipc/context_ipc_tmpl_device.hpp#L366

#define BITWISE_OP(M, inner) \
    M(inner, or) M(inner, and) M(inner, xor)

#define ARITHM_OP(M, inner) \
    M(inner, sum) M(inner, min) M(inner, max) M(inner, prod)

#define FLOAT_TYPES(M, inner) \
    M(float, inner) M(double, inner)
#define INT_TYPES(M, inner) \
    M(short, inner) M(int, inner) M(long, inner) M(longlong, inner)

#define CHAR_TYPES(M, inner) \
    M(char, inner)

#define EXPAND_float(M, Op) M(float, Op) 
#define EXPAND_double(M, Op) M(double, Op)
#define EXPAND_short(M, Op) M(short, Op) 
#define EXPAND_int(M, Op) M(int, Op)
#define EXPAND_long(M, Op) M(long, Op)
#define EXPAND_longlong(M, Op) M(longlong, Op)

#define FOR_FLOAT_OPS(T, inner) \
    ARITHM_OP(EXPAND_##T, inner)
#define INT_FLOAT_OPS(T, inner) \
    ARITHM_OP(EXPAND_##T, inner) \
    BITWISE_OP(EXPAND_##T, inner)

enum class ReduceType {
  sum_, min_, max_, prod_, or_, and_, xor_
};

template < class T, ReduceType Op >
struct AllReduceOp;

template < class Type, ReduceType Op >
__global__ void allreduce_kernel(rocshmem_team_t *teams, Type *dest, 
                const Type *source, size_t total, size_t elems_per_block) {
  __shared__ rocshmem_ctx_t ctx;

  int64_t ctx_type = 0;
  auto thid = threadIdx.x, bid = blockIdx.x;
  rocshmem_wg_team_create_ctx(teams[bid], ctx_type, &ctx);

  for (size_t ofs = blockIdx.x * elems_per_block; ofs < total; 
            ofs += static_cast< size_t >(gridDim.x) * elems_per_block) {

    auto sz = std::min(total - ofs, elems_per_block);
    AllReduceOp< Type, Op >()(ctx, teams[bid], dest + ofs, source + ofs, sz);
  }
  rocshmem_ctx_quiet(ctx);
  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

template < class T >
struct P2POpNbi;

template < class Type >
__global__ void putmem_nbi_kernel(int peer, Type *dest, 
                const Type *source, size_t nelems) {
  
  __shared__ rocshmem_ctx_t ctx;
  int64_t ctx_type = 0;

  rocshmem_wg_ctx_create(ctx_type, &ctx); 
  P2POpNbi< Type >().put(ctx, dest, source, nelems, peer);

  rocshmem_ctx_quiet(ctx);
  __syncthreads();

  rocshmem_wg_ctx_destroy(&ctx);
}

template < class Type >
__global__ void getmem_nbi_kernel(int peer, Type *dest, 
                const Type *source, size_t nelems) {
  
  __shared__ rocshmem_ctx_t ctx;
  int64_t ctx_type = 0;

  rocshmem_wg_ctx_create(ctx_type, &ctx);
  P2POpNbi< Type >().get(ctx, dest, source, nelems, peer);

  rocshmem_ctx_quiet(ctx);
  __syncthreads();

  rocshmem_wg_ctx_destroy(&ctx);
}

__global__ void quiet_kernel(){
  rocshmem_quiet();
}

} // namespace

template < class Type >
absl::Status allreduce_on_stream(rocshmem_team_t *teams, size_t max_teams,
      Type *dest, const Type *source, size_t nreduce, xla::ReductionKind kind, 
                                                    GpuStreamHandle stream) {

  if (stream == nullptr) {
    stream = hipStreamDefault;
  }
  size_t n_threads = 256, elems_per_block = std::min< size_t >(nreduce, 4<<10),
        n_blocks = std::min(nreduce / elems_per_block, max_teams); 
  //fprintf(stderr, "all-reduce nelems: %zd n_blocks: %zd\n", nreduce, n_blocks);

#define LAUNCH(Op) {                                                       \
        allreduce_kernel< Type, ReduceType::Op##_ >                        \
                    <<<dim3(n_blocks), dim3(n_threads), 0, stream>>>       \
                    (teams, dest, source, nreduce, elems_per_block);       \
        break; }

  switch (kind) {
  case xla::ReductionKind::SUM: LAUNCH(sum)
  case xla::ReductionKind::MIN: LAUNCH(min)
  case xla::ReductionKind::MAX: LAUNCH(max)
  case xla::ReductionKind::PRODUCT: LAUNCH(prod)
  default:
    return absl::InternalError("Unsupported reduction type!");
  }
  return absl::OkStatus();
#undef LAUNCH
}

#define DEV_ALL_REDUCE_OP(T, Op)                                               \
template<> struct AllReduceOp< T, ReduceType::Op##_ > {                        \
  __device__ __forceinline__ int operator()(rocshmem_ctx_t ctx,                \
        rocshmem_team_t team, T *dst, const T *src, size_t num) {              \
    return rocshmem_ctx_##T##_##Op##_reduce_wg(ctx, team, dst, src, num);      \
  }                                                                            \
};
FLOAT_TYPES(FOR_FLOAT_OPS, DEV_ALL_REDUCE_OP)
INT_TYPES(INT_FLOAT_OPS, DEV_ALL_REDUCE_OP)

#define ALL_REDUCE_HOST(T, _)                                                  \
template absl::Status allreduce_on_stream< T >(                                \
       rocshmem_team_t *teams, size_t max_teams, T *dest, const T *source,     \
       size_t nreduce, xla::ReductionKind kind, GpuStreamHandle stream);
FLOAT_TYPES(ALL_REDUCE_HOST, ?)
INT_TYPES(ALL_REDUCE_HOST, ?)

template < class Type >
absl::Status get_nbi_on_stream(int peer, Type *dest, 
                const Type *source, size_t nelems, GpuStreamHandle stream) {
  if (stream == nullptr) {
    stream = hipStreamDefault;
  }
  const uint32_t n_blocks = 1, n_threads = 256;
  getmem_nbi_kernel< Type ><<<dim3(n_blocks), dim3(n_threads), 0, stream>>>
       (peer, dest, source, nelems);

  return absl::OkStatus();
}

template < class Type >
absl::Status put_nbi_on_stream(int peer, Type *dest,
                const Type *source, size_t nelems, GpuStreamHandle stream) {
  if (stream == nullptr) {
    stream = hipStreamDefault;
  }
  const uint32_t n_blocks = 1, n_threads = 256;
  putmem_nbi_kernel< Type ><<<dim3(n_blocks), dim3(n_threads), 0, stream>>>
      (peer, dest, source, nelems);

  return absl::OkStatus();
}

#define DEV_P2P_NBI(T, _)                                                  \
template<> struct P2POpNbi< T > {                        \
  __device__ __forceinline__ void get(rocshmem_ctx_t ctx,                \
        T *dst, const T *src, size_t num, int pe) {     \
    rocshmem_ctx_##T##_get_nbi_wg(ctx, dst, src, num, pe);      \
  }                                                                            \
  __device__ __forceinline__ void put(rocshmem_ctx_t ctx,                \
        T *dst, const T *src, size_t num, int pe) {     \
    rocshmem_ctx_##T##_put_nbi_wg(ctx, dst, src, num, pe);      \
  }                                                                            \
};                                                                             \
template absl::Status put_nbi_on_stream< T >(             \
      int peer, T *dest, const T *source, size_t nelem,          \
      GpuStreamHandle stream);      \
template absl::Status get_nbi_on_stream< T >(             \
      int peer, T *dest, const T *source, size_t nelem,          \
      GpuStreamHandle stream);

FLOAT_TYPES(DEV_P2P_NBI, ?)
INT_TYPES(DEV_P2P_NBI, ?)
CHAR_TYPES(DEV_P2P_NBI, ??)

void rocshmem_quiet_on_stream(hipStream_t stream) {
  if (stream == nullptr) {
    stream = hipStreamDefault;
  }
  quiet_kernel<<<1, 1, 0,  stream>>>();  
}

void synchronize_all() {
  hipDeviceSynchronize();
}

} // namespace rocshmem
