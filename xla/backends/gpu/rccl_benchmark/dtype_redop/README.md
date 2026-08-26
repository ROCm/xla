# `dtype_redop/` - element types and reduction operators

**Status: not implemented.** This file describes what belongs here.

## The mechanism

Every collective carries an element type and, for reductions, an operator. RCCL
instantiates device kernels per (type, operator) pair, so the matrix here is a
matrix of separate kernels rather than of arguments to one.

XLA maps its own types onto RCCL's in `rccl_types.cc`, and that mapping is
architecture-dependent: the fp8 formats are accepted only on parts that support
them, checked against `rocm_compute_capability`. A mapping error is a whole
element type silently interpreted as another one.

Types XLA maps: bf16, fp16, fp32, fp64, the signed and unsigned integer widths,
and fp8 subject to the architecture check.
Operators: `ncclSum`, `ncclProd`, `ncclMin`, `ncclMax`.

## Why XLA reaches it

By ordinary use - and one case is reachable without anyone asking for it:
`GpuAllGatherCombiner` is constructed with `combine_different_dtypes=true`, and
`CollectiveConfig` carries one element type per buffer, so **a single combined
AllGather can carry buffers of different types**. That is a per-buffer type
mapping exercised inside one group, which no single-collective test reaches.

## Planned cases

- One pass per (type, operator) pair the mapping supports, at a size that stays
  in one protocol bucket so this directory is not accidentally re-testing
  `protocol_selection/`.
- fp8 gated on architecture, and asserted to be **rejected** where unsupported
  rather than silently accepted.
- Mixed-dtype buffers in one combined AllGather, which belongs to
  `group_plan/` by shape and here by type; the case lives in `group_plan/` and
  is cross-referenced from this file.
- Reductions need a tolerance model rather than bit-exact comparison, because
  the summation order is not fixed. Movement-only collectives stay bit-exact.
  The distinction should be explicit per case, not a global relaxation - a
  tolerance wide enough for fp16 AllReduce would hide real corruption in an
  AllGather.

## Why this is P2 and still worth doing

Nothing here is subtle, and none of it is likely to break on its own. It is
cheap, and it is the coverage that catches a whole element type quietly
disappearing after a mapping change - the kind of defect that is trivial to find
once suspected and easy to ship if nobody looks.

## Source references

| What                  | Where                                                        |
| --------------------- | ------------------------------------------------------------ |
| Type mapping          | `xla/backends/gpu/collectives/rccl_types.cc`                  |
| Per-buffer types      | `CollectiveConfig::operand_element_type`                      |
| Mixed-dtype combining | `xla/service/gpu/gpu_compiler.cc`, `AddCollectiveCombinerPasses` |
