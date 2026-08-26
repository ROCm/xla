# `warp_speed/` - feature activation and multi-task kernel plans

## The mechanism

WarpSpeed is an RCCL execution mode that maps several logical channels onto one
GPU block: a block's warps cover `warpCount` consecutive channels, and a warp's
global identity is

```
globalWarpId = warpCount * blockIdx.x + localWarpId
```

Separately, RCCL assigns each task in a kernel plan its own contiguous range of
channels (`src/enqueue.cc`, `devWork->channelLo` / `channelHi`), advancing the
channel cursor from one task to the next and growing `plan->nWorkBatches` as it
goes.

These two facts interact. The kernel loads the block's initial work batch by
block index:

```c
loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
```

With one task per plan, block index and batch index agree and nothing is wrong.
With more than one task, the channel cursor has moved on while the batch index
has not, so a block whose warps serve a later task's channels loads an earlier
task's work descriptor - and then executes it: wrong pointers, wrong element
counts, reads and writes outside the buffers it was given.

**One collective never exposes this. Two in the same group do.**

## Why XLA reaches it

Not by accident, and not only in unusual configurations:

- `AllGatherThunk` submits **every buffer it owns inside a single
  `GroupExecute`** (`xla/backends/gpu/runtime/all_gather_thunk.cc`,
  `RunAllGather`). Buffer count is task count.
- The collective combiner packs buffers into one collective up to a byte
  threshold, with a count limit of 2048
  (`xla/service/collective_utils.h`). A combined collective is by construction a
  multi-task plan.
- On ROCm, AllToAll and CollectivePermute are built from grouped `ncclSend` /
  `ncclRecv` pairs (`rccl_communicator.cc`, `LaunchAllToAll`), so an eight-rank
  AllToAll is sixteen operations in one group with no configuration at all.

## Reachability, and why the default configuration is not enough

Activation requires all of:

| Condition                | Source                                             |
| ------------------------ | -------------------------------------------------- |
| built with WarpSpeed     | `ENABLE_WARP_SPEED`, which CMake defaults to `OFF` |
| `RCCL_WARP_SPEED_AUTO`   | non-zero; the default has differed between versions |
| architecture             | gfx950 in auto mode                                |
| single node              | `comm->nNodes == 1`                                |
| Ring algorithm selected  | other algorithms return early                      |
| transfer size            | at or above the activation threshold                |

The size condition is on the **aggregate traffic of the kernel plan**, not on
one operand and not on one collective, and the threshold is per collective type:

```
RCCL_PARAM(WarpSpeedAGThreshold, "WARP_SPEED_AG_THRESHOLD", 134217728);  // 128 MiB
RCCL_PARAM(WarpSpeedARThreshold, "WARP_SPEED_AR_THRESHOLD",  67108864);  //  64 MiB
RCCL_PARAM(WarpSpeedRSThreshold, "WARP_SPEED_RS_THRESHOLD", 2147483648); //   2 GiB
```

Another checkout of the same feature has no such tunables at all and a single
hard-coded 64 MiB constant, with `WARP_SPEED_AUTO` defaulting to 0 rather than
1. Two checkouts, two threshold mechanisms, opposite defaults - which is the
whole argument for `extract_thresholds.py` generating these numbers instead of
the cases carrying them.

Measured against the reference library, matching the 128 MiB AllGather
threshold above:

| Plan                                   | Aggregate | Activated |
| -------------------------------------- | --------- | --------- |
| 1 AllGather, 8 MiB/rank, 8 ranks        | 64 MiB    | no, logged as below threshold |
| 1 AllGather, 16 MiB/rank, 8 ranks       | 128 MiB   | yes       |
| 2 AllGathers, 8 MiB/rank, 8 ranks       | 128 MiB   | yes       |
| 2 AllGathers, 4 MiB/rank, 8 ranks       | 64 MiB    | no, logged as below threshold |

This matters for how the arms are sized. Comparing a two-buffer case against a
one-buffer case at the same per-rank size would put the one-buffer case below
the threshold, so it would pass by never entering the branch - a control that
controls for nothing. The arms are matched on aggregate traffic instead.

There is a second consequence worth stating plainly. XLA's default AllGather
combiner threshold is 30 MiB of output, so a combined collective built under
stock defaults carries an aggregate far below the activation threshold. **Under
stock XLA defaults this branch is unreachable**; it took a workload configured
with a much larger combiner threshold to get there. A lane that runs only stock
defaults would exercise none of this and report a pass for it.

The arms therefore include the production-style configuration, and treat the
lowered-threshold arm as an addition rather than a substitute: a lane built only
on overridden thresholds keeps passing after the library changes the thresholds
it ships with.

## Reading the log: availability is not activation

Two messages look alike and mean different things.

- `WarpSpeed enabled: warpSpeedChannelMultiplier ...` is emitted **per
  communicator** and only says the feature is compiled in and switched on. It
  appears even for transfers the library then declines to use it for.
- `RCCL Warp Speed Channels set to N. Warps per block is set to M` comes from
  the path taken when a collective **actually runs** under the feature. This is
  the activation signal.

Treating the first as activation makes a case that never entered the branch
indistinguishable from one that entered it and behaved. `common/path_assert.h`
keys `warp_speed_active` on the second and records the first separately as
`warp_speed_available`.

## Cases and observed results

`grouped_all_gather_test.cc`, one arm per process (`run.sh case <arm>`).

Measured against `rocm/jax-training:maxtext-v26.5`
(ROCm 7.14.0, RCCL 2.30.4-HEAD:9e5e408, 8x MI355X / gfx950, tuning model 6):

| Arm                           | Tasks/plan | Aggregate | WarpSpeed | Result       |
| ----------------------------- | ---------- | --------- | --------- | ------------ |
| `grouped_two_on`              | 2          | 128 MiB   | **active**    | **fault**    |
| `single_on`                   | 1          | 128 MiB   | active    | pass         |
| `separate_groups_on`          | 1 per plan | 128 MiB   | active    | pass         |
| `grouped_two_below_threshold` | 2          | 64 MiB    | declined  | pass         |
| `grouped_two_feature_off`     | 2          | 128 MiB   | off       | pass         |
| `grouped_four_on`             | 4          | 256 MiB   | **active**    | **fault**    |
| `grouped_two_forced_small`    | 2          | 16 MiB    | **forced active** | **mismatch** |

Every arm's WarpSpeed column is asserted against the library's own log, not
assumed: the three failing arms each logged eight activations (one per rank),
`grouped_two_below_threshold` logged the decline at 64 MiB, and
`grouped_two_feature_off` logged nothing.

The separation is exact. **More than one task in a plan while WarpSpeed is
active** is the only combination that fails; matching the traffic, the feature
state, or the operation count individually is not enough to reproduce it.

`separate_groups_on` is the sharpest of the controls: identical buffers,
identical sizes, WarpSpeed active for both collectives (sixteen activation
lines), and it passes. Only the submission differs.

## Two failure shapes, both needed

`grouped_two_on` **faults**: the misrouted descriptor addresses memory far
enough away to trip a hardware memory fault, killing the process with

```
Memory Fault Error [... kernel: ncclDevKernel_Generic_2(ncclDevKernelArgsStorage<5120ul>)]
```

which is the same signature the standalone RCCL reproducer produced.

`grouped_two_forced_small` **corrupts silently instead**: no fault, no crash,
1,446,893 of 4,194,304 output words wrong, beginning exactly at the boundary of
rank 2's contribution. The per-buffer, per-rank payload makes the corruption
self-describing - the decoded mismatches name buffers 3 and 7, which do not
exist in a two-buffer case, alongside bytes carrying the guard poison value:

```
word 1048576: got <foreign 0xbaa5>, expected buffer=0 rank=2 tag=0x00
word 1048578: got buffer=3 rank=6 tag=0xd9, expected buffer=0 rank=2 tag=0x02
word 1048579: got buffer=7 rank=4 tag=0x1f, expected buffer=0 rank=2 tag=0x03
```

A payload that only encoded position would have shown "wrong number here". This
one says where the data came from, which is the difference between knowing
something broke and knowing what broke.

This is why the oracle needs both a fault classifier and guard regions, and why
each arm runs in its own process: a fault in one arm must not take the sweep
with it.

## When the feature does not activate

Every arm reports **inconclusive** rather than passing. Against a library
without WarpSpeed compiled in - which includes the RCCL shipped in most ROCm
releases, and the hermetic ROCm that bazel fetches - that is the correct and
expected result, and it is the reason the library under test must be injected
rather than assumed.

## Source references

| What                          | Where                                                          |
| ----------------------------- | -------------------------------------------------------------- |
| Activation conditions         | `rccl/src/rccl_wrap.cc`, `rcclSetWarpSpeedAuto`                 |
| Channel halving when active   | `rccl/src/enqueue.cc`                                           |
| Task to channel-range mapping | `rccl/src/enqueue.cc`, `devWork->channelLo` / `channelHi`       |
| Work batch load               | `rccl/src/device/common.h`, `ncclKernelMain`                    |
| Build option                  | `rccl/CMakeLists.txt`, `option(ENABLE_WARP_SPEED ... OFF)`      |
| Buffers to group operations   | `xla/backends/gpu/runtime/all_gather_thunk.cc`, `RunAllGather`  |
| Combiner limits               | `xla/service/collective_utils.h`                                |

## Related

The grouping dimension itself - operation count, mixed collectives in one
group, mixed dtypes, multiple communicators - belongs in `group_plan/` and is
swept there across feature states. This directory owns the pinned regression for
the known defect; `group_plan/` owns the question of whether grouping breaks
anything else.
