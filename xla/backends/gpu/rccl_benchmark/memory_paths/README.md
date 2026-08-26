# `memory_paths/` - registered, symmetric and plain buffers

**Status: not implemented.** This file describes what belongs here.

## The mechanism

How a buffer is made known to RCCL changes how the transfer is performed, not
just how it is set up:

| Path                       | API                                              |
| -------------------------- | ------------------------------------------------ |
| Private (default)          | none; both sides exchange local pointers          |
| Registered buffer          | `ncclCommRegister` / `ncclCommDeregister`         |
| Library-allocated          | `ncclMemAlloc` / `ncclMemFree`                    |
| Symmetric window           | `ncclCommWindowRegister`, `ncclWindow`            |

The symmetric path is the sharpest break: it enables one-sided transfers, where
a rank writes directly into a peer's buffer rather than both sides
rendezvousing. That is a different set of device kernels and a different
synchronization protocol, so nothing established about the two-sided path
carries over.

## Why XLA reaches it

XLA calls all of these
(`xla/backends/gpu/collectives/rccl_{registered,symmetric}_memory.cc`), and
`CollectivesMode` selects between them per collective:
`COLLECTIVES_PRIVATE_MEMORY` (the default), `COLLECTIVES_SYMMETRIC_MEMORY` and
`COLLECTIVES_PEER_MEMORY` (`xla/xla.proto`).

There is also a one-sided AllGather implemented directly in XLA
(`all_gather_thunk.cc`, `RunOneSidedAllGather`: signal, wait, put, wait) which
bypasses the collective entirely when symmetric memory is in use. It is worth
covering here for the same reason RCCL's own symmetric path is: it is a
different implementation reaching the same answer.

## Planned cases

- The same shapes as `group_plan/` run once per memory path, so a defect can be
  attributed to the path rather than to the shape.
- Registered versus unregistered with identical buffers, since registration is
  meant to be transparent to the result.
- Symmetric window with the one-sided AllGather, checked against the two-sided
  result bit for bit.
- Registration lifetime: register, run, deregister, run again; and buffers
  registered on one communicator used from another.
- Guard regions are especially worth having here. A registration path that maps
  the wrong extent is exactly the failure these detect.

## Source references

| What                     | Where                                                        |
| ------------------------ | ------------------------------------------------------------ |
| Registered memory        | `xla/backends/gpu/collectives/rccl_registered_memory.cc`      |
| Symmetric memory         | `xla/backends/gpu/collectives/rccl_symmetric_memory.cc`       |
| One-sided AllGather      | `xla/backends/gpu/runtime/all_gather_thunk.cc`                |
| Mode selection           | `xla/xla.proto`, `CollectivesMode`                            |
