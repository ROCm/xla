# `group_plan/` - what shares one RCCL kernel plan

**Status: not implemented.** This file describes what belongs here.

## The mechanism

RCCL builds a kernel plan per group and gives each task in it its own
contiguous range of channels, advancing the channel cursor from one task to the
next (`rccl/src/enqueue.cc`, `devWork->channelLo` / `channelHi`,
`plan->nWorkBatches`).

Everything about how a plan is composed - how many tasks, whether they are the
same operation, whether they share a communicator - is decided on the host and
then baked into structures the device kernel indexes into. That indexing is
where multi-task plans have gone wrong before, and this directory is where the
composition itself is swept.

**This is the dimension `rccl-tests` structurally does not have.** It submits
one collective at a time, so no amount of running it explores this axis.

## Why XLA reaches it

Three mechanisms stack, none of them opt-in:

1. **The combiner.** `AllGatherThunk` submits every buffer it owns inside one
   `GroupExecute` (`xla/backends/gpu/runtime/all_gather_thunk.cc`), and the
   collective combiner packs buffers into one collective up to a byte threshold
   with a **count limit of 2048** (`xla/service/collective_utils.h`). Buffer
   count is task count.
2. **`GroupCollectivesByKey`.** Runs unconditionally in the pipeline
   (`xla/service/gpu/gpu_compiler.cc`), groups instructions carrying a
   `collective_group_key` attribute, and its default predicate is AllGather +
   ReduceScatter + AllReduce. Its own comment names FSDP layers as the intended
   use. The result is a `CollectiveGroupThunk`, which can also issue a
   **multi-communicator** group launch.
3. **Grouped Send/Recv.** On ROCm, AllToAll and CollectivePermute are built from
   `ncclSend`/`ncclRecv` pairs (`rccl_communicator.cc`, `LaunchAllToAll`), so an
   eight-rank AllToAll is **sixteen operations in one group** with no
   configuration at all.

## Planned cases

| Axis                        | Values                                                   |
| --------------------------- | -------------------------------------------------------- |
| Operations per plan         | 1, 2, 8, 64, 512, 2048                                   |
| Homogeneity                 | same op / same op mixed dtypes / AllGather+ReduceScatter+AllReduce mixed |
| Communicators               | one / several (`CollectiveGroupThunk` multi-comm path)   |
| Channel-range layout        | all tasks in one bucket / tasks spanning different buckets |
| Send/Recv shapes            | 8-rank AllToAll, CollectivePermute rings, half-exchanges, ranks with no peer |
| Feature context             | swept with WarpSpeed on and off, so a finding is attributable |

Mixed dtypes are reachable by default: `GpuAllGatherCombiner` is constructed
with `combine_different_dtypes=true`, and `CollectiveConfig` carries one element
type per buffer, so a thunk can express it directly.

## Relationship to `warp_speed/`

`warp_speed/` owns the pinned regression for the one defect already known at
this intersection. This directory owns the general question: **does sharing a
plan break anything else**, under any feature state. The overlap is one point
and is intentional.

## Source references

| What                          | Where                                                          |
| ----------------------------- | -------------------------------------------------------------- |
| Task to channel-range mapping | `rccl/src/enqueue.cc`                                           |
| Buffers to group operations   | `xla/backends/gpu/runtime/all_gather_thunk.cc`, `RunAllGather`   |
| Heterogeneous grouping        | `xla/backends/gpu/transforms/group_collectives_by_key.{h,cc}`    |
| Multi-communicator launch     | `xla/backends/gpu/runtime/collective_group_thunk.cc`             |
| Combiner limits               | `xla/service/collective_utils.h`                                 |
| Grouped Send/Recv             | `xla/backends/gpu/collectives/rccl_communicator.cc`              |
