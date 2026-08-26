# `comm_lifecycle/` - creating, splitting, aborting and rebuilding communicators

**Status: not implemented.** This file describes what belongs here.

## The mechanism

Communicator setup and teardown is a large amount of RCCL that the data-plane
cases never touch, and it is where a long-running job spends its worst moments:
startup, reconfiguration, and recovery from a failed step.

| Area              | API                                                                     |
| ----------------- | ----------------------------------------------------------------------- |
| Creation          | `ncclCommInitRank`, `ncclCommInitRankConfig`, `ncclCommInitRankScalable`, `ncclCommInitAll` |
| Sub-communicators | `ncclCommSplit`                                                          |
| Non-blocking mode | `ncclCommGetAsyncError`, polling on `ncclInProgress`                     |
| Teardown          | `ncclCommDestroy`, `ncclCommAbort`                                       |

`ncclCommSplit` deserves particular attention: a split communicator has its own
rank ordering, its own channel layout and its own tuning decisions, so every
size-dependent conclusion from the full communicator has to be re-established
on the derived one.

## Why XLA reaches it

XLA calls all four creation entry points and both teardown paths, and its clique
machinery genuinely splits communicators rather than only creating world-sized
ones. Non-blocking initialization with `ncclInProgress` polling is on the normal
startup path, and `ncclCommAbort` is on the failure path a training job takes
when a step goes wrong - which is the moment when a communicator bug is least
welcome and least likely to have been tested.

## Planned cases

- Split 8 ranks into 2x4 and 4x2; run the `group_plan/` shapes on the derived
  communicators; confirm from the log that the tuning decisions were re-made for
  the new size rather than inherited.
- Concurrent collectives on several communicators derived from the same parent.
- Non-blocking initialization driven to completion through `ncclInProgress`,
  including a rank that is slow to arrive.
- Destroy and recreate in a loop, running a collective each time. Communicator
  reuse is where stale state shows up.
- Abort mid-collective, then rebuild and run. The check is that recovery works,
  not merely that abort returns.
- Repeated execution on one communicator, since some corruption only appears
  after a communicator has been reused.

## A note on failure classification

Cases here are more likely to hang than to produce wrong data, so the watchdog
and the timeout classification in `run.sh` carry more weight than the payload
oracle. A hang reported as a timeout is a result; a hang that stalls the sweep
is not.

## Source references

| What                    | Where                                                    |
| ----------------------- | -------------------------------------------------------- |
| Communicator lifecycle  | `xla/backends/gpu/collectives/rccl_communicator.cc`       |
| Clique creation         | `xla/backends/gpu/collectives/gpu_cliques.cc`             |
| Clique keys and splits  | `xla/backends/gpu/collectives/gpu_clique_key.cc`          |
