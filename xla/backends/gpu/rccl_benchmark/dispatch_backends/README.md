# `dispatch_backends/` - which implementation runs at all

**Status: not implemented.** This file describes what belongs here.

## The mechanism

Before any of the algorithm, protocol or channel decisions, RCCL picks **which
implementation** handles the call. These are not variations of one code path;
they are separate implementations selected by size and by configuration.

| Backend                    | Selected when                                                       | Default                     | Source                                   |
| -------------------------- | ------------------------------------------------------------------- | --------------------------- | ---------------------------------------- |
| MSCCL (algorithm plugin)   | `mscclAvailable(comm)`, covering AllGather / AllToAll / AllToAllv / AllReduce | depends on the XML present | `rccl/src/collectives.cc`                |
| MSCCL++                    | `RCCL_MSCCLPP_ENABLE` and size <= threshold                          | disabled, threshold 16 MiB  | `rccl/src/init.cc`                       |
| rocSHMEM GDA AllToAll      | `rcclUseAllToAllGda(comm)` and size <= threshold                     | **enabled**, threshold 256 KiB | `rccl/src/init.cc`, `collectives.cc`  |
| Pivot AllToAll             | `topo->pivotA2AEnabled` and enough channels                          | **enabled**                 | `rccl/src/init.cc`, `collectives.cc`     |
| Standard path              | none of the above matched                                            |                             |                                          |

Two of these are on by default, so a workload can be dispatched away from the
standard path without anyone configuring anything - and a small-message AllToAll
test that "passes" may never have touched the code a large one uses.

## Why XLA reaches it

The two default-enabled backends are reachable with no configuration: both key
off AllToAll, which on ROCm is XLA's implementation for AllToAll and
CollectivePermute. Message size alone decides.

MSCCL and MSCCL++ need to be switched on, so they are lower priority - but they
are also the two most likely to appear in a future release enabled by default,
and `extract_thresholds.py` will flag the default changing.

## Planned cases

- AllToAll at `{256 KiB - 1, 256 KiB, 256 KiB + 1}` per rank, asserting from the
  log which backend handled it. Crossing this boundary silently changes the
  implementation under test, so the assertion matters more than the result.
- Pivot AllToAll at channel counts above and below `pivotA2ANumBiRings * 2`.
- MSCCL++ enabled explicitly, straddling its 16 MiB threshold.
- MSCCL with a representative XML, once there is one worth pinning.
- Each backend swept against the correctness oracle unchanged. The point is that
  a different implementation gets held to the same standard, not a weaker one.

## A note on attribution

Cases here answer "was the right implementation selected, and is it correct".
Once inside the standard path, protocol and channel questions belong in
`protocol_selection/` and `channel_buckets/`. If a case fails and the backend
selection was as expected, it belongs in one of those instead.

## Source references

| What                      | Where                                     |
| ------------------------- | ----------------------------------------- |
| Backend dispatch          | `rccl/src/collectives.cc`                  |
| Thresholds and enables    | `rccl/src/init.cc`                         |
| AllToAll on the XLA side  | `xla/backends/gpu/collectives/rccl_communicator.cc` |
| Generated thresholds      | `../thresholds.json`                       |
