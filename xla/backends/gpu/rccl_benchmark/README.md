# XLA-driven RCCL benchmark and regression suite

The question this suite answers:

> For the RCCL call shapes XLA actually produces, is a given `librccl.so` build
> correct, stable, and free of obvious performance regressions?

**The subject under test is RCCL. XLA is the driver.** That inversion is the
whole design, and it is what separates this from the collective tests already in
`xla/backends/gpu/runtime`:

|                    | `*_thunk_multigpu_test`      | this suite                      |
| ------------------ | ---------------------------- | ------------------------------- |
| Subject / fixture  | XLA is tested, RCCL assumed  | **RCCL is tested, XLA assumed** |
| Variable dimension | XLA commit                   | **RCCL commit / ROCm version**  |
| Coverage target    | XLA code paths and arguments | **RCCL internal branches**      |
| On failure         | fix XLA                      | **produce a repro for AMD**     |
| Runs when          | XLA changes                  | an RCCL build changes           |
| Scale              | 2-4 GPUs, seconds            | 8 GPUs, minutes                 |

The suite reuses the multi-GPU scaffolding from
`xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h`, but it is not
part of the XLA unit tests and must not be added to their lane: it needs eight
GPUs, large transfers, a specific library configuration, and minutes rather than
seconds.

The coverage question is therefore never "does XLA have this code path" but
**"does RCCL take a different branch because of this"**. Those give different
answers. The defect this suite was built around looks entirely ordinary from
XLA's side - valid arguments, normal code path - and depends only on a transfer
size crossing a boundary XLA knows nothing about.

## Layout

Subdirectories are named after **the RCCL mechanism under test** - the thing
that would be broken - and not after a collective. Collective type, dtype and
size are parameters *within* a directory.

Organizing by collective would leave the cross-cutting mechanisms homeless:
threshold sweeps would be copied into six near-identical directories, and shapes
that span collectives (several different collectives in one group) would have
nowhere to live at all.

```
rccl_benchmark/
├── common/                 shared infrastructure; every case uses it
├── extract_thresholds.py   generates the matrix inputs from library source
├── run.sh                  builds and runs cases, one arm per process
├── warp_speed/             feature activation and its effect on plan execution
└── ...                     one directory per mechanism as coverage grows
```

Planned siblings, in the order they matter:
`group_plan/` (operations per group, homogeneous vs mixed, single vs multiple
communicators), `protocol_selection/` (LL / LL128 / Simple boundaries),
`channel_buckets/` (the nine per-rank size buckets), `dispatch_backends/`
(MSCCL, MSCCL++, rocSHMEM, pivot AllToAll), `memory_paths/`,
`comm_lifecycle/`, `dtype_redop/`.

**Attribution rule.** A case belongs to the directory of the mechanism most
likely to be at fault if it fails. Cases that sit at an intersection are
cross-referenced from the other directory's README, never copied.

## Rules that are not optional

1. **Everything shared lives in `common/`.** Eight directories with eight
   slightly different notions of "the payload is correct" would be eight
   different standards, and then a result would not mean the same thing
   depending on where it came from.
2. **Every mechanism directory has a README** naming the mechanism, the RCCL
   source it corresponds to, any known defect, and whether XLA can reach it by
   default. Keeping that next to the code is the only version of it that stays
   true.
3. **Recorded HLO carries a manifest** with the XLA commit, the flags and the
   workload it came from. Post-optimization HLO is a dated snapshot; without
   provenance nobody will dare touch it in six months.

## Running

```bash
./run.sh build                       # build inside the container
./run.sh list                        # available arms
./run.sh case grouped_two_on         # one arm, one process
./run.sh matrix 3                    # every arm, three repeats each
RCCL_PREFIX=/path/to/rccl ./run.sh matrix   # against an injected library
```

Results land in `results/`, one log per arm plus `summary.tsv`. Each log opens
with a manifest: image id, XLA commit, ROCm version, GPU count, and the
identity of the RCCL that was loaded.

**Do not run these under `--config=ci_multi_gpu`.** That config pins
`NCCL_MAX_NCHANNELS=1`, exposes four GPUs via `HIP_VISIBLE_DEVICES=0,1,2,3`, and
sets `--flaky_test_attempts=3`. Each of those three, on its own, is enough to
hide the class of defect this suite exists to find; the first makes an entire
category of channel-assignment bug structurally unreachable.

## What makes a result trustworthy

Correct output is not evidence that anything was tested. Most of the design
budget here goes to closing the ways a case can report a pass it has not earned:

- **Path assertion** (`common/path_assert.h`). Each arm declares what the
  library should do and checks it against the debug log. A build where the
  targeted feature never activates fails as inconclusive rather than passing.
  This is not hypothetical: the feature is compiled out of most release builds,
  disabled by default when present, and its default flipped between versions.
- **Guard regions** (`common/guarded_buffer.h`). Payloads are padded with
  position-dependent poison. An out-of-bounds write is reported as
  "overwritten N bytes past the end of this buffer" instead of surfacing as a
  NaN several steps downstream.
- **Per-buffer, per-rank payloads** (`common/data_pattern.h`). Every word
  encodes which buffer and which rank produced it, so two same-shaped
  collectives that swap or share their data are caught. With a position-only
  pattern they would compare equal.
- **Sources are verified too.** A collective must not modify its input; a task
  executing through the wrong descriptor can corrupt one.
- **Insufficient hardware fails.** A case that cannot run on the GPUs present
  reports a failure, not a skip, because a skipped case reads as coverage.
- **The loaded library is identified**, by path and build id, not merely
  assumed from a file that exists somewhere on the search path.

## Sizes come from the library, not from round numbers

Sizes are chosen to straddle the boundaries RCCL branches on, and those
boundaries are read out of the library source by `extract_thresholds.py` rather
than written into the tests:

```bash
./extract_thresholds.py --rccl-src ~/repos/rccl_src --xla-src ~/repos/xla \
    -o thresholds.json
./extract_thresholds.py --rccl-src ~/repos/rccl_src --diff thresholds.json
```

`--diff` exits non-zero when anything moved. That is not an error in itself -
it means the library reorganized its decisions and the matrix needs revisiting
before its results can be believed again. The same report also lists the RCCL
entry points XLA calls, so a new one shows up as coverage that does not exist
yet.

Round sizes are avoided deliberately: 1 MiB, 16 MiB and 256 MiB can all sit on
the same side of every threshold that matters, producing a matrix that looks
broad and tests one branch.

## Environment constraints

Each of these was found by a case failing to report anything useful, and each
one, left unaddressed, produces a green result that means nothing.

- **Build with `--dynamic_mode=off`.** `run.sh build` does. XLA's default ROCm
  configuration (`--config=rocm`) uses a hermetic toolchain; the resulting
  dynamically linked binary loads the container's `libLLVM` alongside its own
  and crashes before `main`.
- **Build against the container's ROCm**, i.e. `build --config rocm_clang_local`
  in `xla_configure.bazelrc`, not the hermetic default. Otherwise executor
  initialization reaches hipBLASLt built against a different C++ standard
  library and segfaults inside `std::filesystem::path`. This is also why only
  RCCL is ever injected at runtime, never a whole foreign ROCm prefix: RCCL's
  interface is C, so it can be swapped safely.
- **Pass the numeric group ids of `/dev/kfd` and `/dev/dri/*`.** `--group-add
  render` resolves against the container's group table, which does not match the
  host's. Without it the runtime reports zero devices - and a suite that skips
  on missing hardware would report a pass.
- **The library's environment must be set before the process starts.** RCCL
  snapshots it at load time, so `setenv` from `main` is too late; the binary
  re-executes itself once to work around callers who do not know that. It is
  also the reason arms that differ in library parameters must be separate
  processes.
- **`NCCL_DEBUG=VERSION` is preset in some images**, and it prints a banner and
  nothing else. The suite overrides `NCCL_DEBUG` unless it is already at INFO or
  TRACE, because respecting it would leave every case unable to confirm its
  path.
- **`NCCL_DEBUG_SUBSYS=ALL` produces no output** with the reference library.
  `INIT,TUNING,COLL` is used instead.

## The failure mode this suite is built to avoid, observed here

While bringing this up, the pre-existing
`all_gather_thunk_multigpu_test_amdgpu_any` was run in a container that could
not see any GPU. It reported `PASSED` to bazel. Its log shows all three of its
cases were skipped for want of GPUs.

That is not a criticism of that test - skipping is reasonable for a unit test
that has to run anywhere. It is the exact reason this suite does the opposite:
insufficient hardware, an unconfirmed code path, and a library that lacks the
feature under test all produce failures here, because coverage that is reported
but not delivered is worse than no coverage.

## A note on the two lenses

A mechanism can be tested through a thunk, through HLO, or both, in the same
directory.

- **Thunk cases** fix the call sequence in C++ constants, so no compiler pass
  can change what reaches RCCL. They cost tight coupling to unstable internal
  APIs - which fails loudly at compile time - and they are the right basis for a
  gate.
- **HLO cases** go through the real compiler, which is the only way to confirm
  that XLA still produces the shapes the thunk cases assume. They couple to
  compiler behaviour instead, and that failure mode is silent: the test stays
  green while testing something else.

Pin the XLA commit and move it deliberately. The library is supposed to be the
only thing changing between runs; a baseline that drifts because the compiler
moved cannot be attributed to anything.
