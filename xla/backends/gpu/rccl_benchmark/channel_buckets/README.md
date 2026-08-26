# `channel_buckets/` - how many channels a transfer gets

**Status: not implemented.** This file describes what belongs here.

## The mechanism

The number of channels a collective runs on is chosen from a per-collective
table keyed by **per-rank** transfer size
(`rccl/src/graph/tuning.cc`, `rcclTuningModel[].channelThresholds`), then
adjusted against a thread threshold (`rccl/src/enqueue.cc`).

Channel count decides how work is split across blocks, so it changes the
parallel decomposition of every collective - and it is the quantity a whole
class of indexing defects is sensitive to, including the one `warp_speed/`
pins down.

For gfx950 (model 6), AllGather:

| Per-rank size          | Channels |
| ---------------------- | -------- |
| [2 KiB, 4 KiB)         | 2        |
| [4 KiB, 8 KiB)         | 4        |
| [8 KiB, 16 KiB)        | 8        |
| [16 KiB, 256 KiB)      | 16       |
| [256 KiB, 512 KiB)     | 32       |
| [512 KiB, 1 MiB)       | 40       |
| -                      | 48 (entry disabled) |
| [1 MiB, 4 MiB)         | 56       |
| [4 MiB, 256 MiB)       | 64       |

ReduceScatter uses the same bucket counts with the ranges shifted down;
AllReduce does not use this table at all.

Note the disabled 48-channel entry: a bucket that exists in the table and can
never be selected. Coverage derived from the table rather than from round
numbers gets that right for free.

## Why XLA reaches it - and where it does not

Partially, and that is itself the finding. XLA's default AllGather combiner
threshold is 30 MiB of output, so per-rank size on eight ranks stays under
4 MiB, which means **the 64-channel bucket is unreachable under stock XLA
defaults**. Reaching it needs a raised combiner threshold, which production
workloads do use.

A lane that ran only stock defaults would report coverage of this table while
never entering its top bucket. Arms here therefore sweep the combiner threshold
as a first-class parameter.

## Planned cases

- `{bucket_lower - 1, bucket_lower, bucket_lower + 1}` for all eight live
  boundaries, for AllGather and ReduceScatter.
- The disabled 48-channel entry asserted as unreachable, so that it becoming
  reachable is noticed.
- Assert the channel count the library actually chose, read from
  `post-adjustment based on threadThreshold:... nc:%i` via
  `common/path_assert.h`. Landing in the intended bucket is the point; getting
  the right answer from the wrong bucket is not coverage.
- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` as diagnostic lanes only. Pinning
  the channel count is what the existing ROCm CI does
  (`NCCL_MAX_NCHANNELS=1`), and it makes this entire table untestable.

## Source references

| What                       | Where                                                    |
| -------------------------- | -------------------------------------------------------- |
| Channel bucket table       | `rccl/src/graph/tuning.cc`, `rcclTuningModel[]`           |
| Channel count adjustment   | `rccl/src/enqueue.cc`                                     |
| Combiner threshold         | `xla/service/collective_utils.h`                          |
| Generated thresholds       | `../thresholds.json`                                      |
