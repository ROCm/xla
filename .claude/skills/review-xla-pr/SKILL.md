---
name: review-xla-pr
description: Review an XLA pull request with deep expertise in HLO optimizations, GPU backend, Triton codegen, autotuner, and AMD/ROCm parity. Use when asked to review an XLA PR or check an XLA change.
argument-hint: [PR-number]
context: fork
agent: general-purpose
allowed-tools: Bash(gh *), Bash(git *), Bash(head *), Bash(grep *), Read, Grep, Glob
---

# XLA PR Review

## Fetching Pull Request Context

Before reviewing, fetch the PR context by running these commands:
- `gh pr view $ARGUMENTS` — PR title, description, author
- `gh pr diff $ARGUMENTS --name-only` — list of changed files
- `gh pr diff $ARGUMENTS` — full diff
- `gh pr view $ARGUMENTS --comments 2>/dev/null || echo "(no comments yet)"` — existing review comments

## Your Task

Review the PR above thoroughly using the checklist below. Be specific: cite file paths and line numbers from the diff. Describe each finding clearly and explain why it matters. Do NOT assign severity labels (no BLOCKER/WARNING/NIT) — let the human reviewer judge importance.

**CRITICAL SCOPE RULE**: Only flag issues that exist in the PR diff itself — lines added or modified by this PR. Do NOT flag pre-existing code that the PR did not touch, even if that code is in the same files. If a pre-existing problem is worth noting, mention it briefly in a separate "Pre-existing issues (out of scope)" section at the end, never as a BLOCKER or WARNING against this PR.

---

## XLA Review Checklist

### 1. HLO IR Correctness
- Does the change preserve the `HloModule` / `HloComputation` / `HloInstruction` invariants?
- If new opcodes or shape semantics are introduced, are all `DfsHloVisitor` `Handle*` methods updated?
- Are `HloPassPipeline` / `HloPassFix` used correctly? Fixed-point passes must not loop infinitely (check `kMaxIterations` = 25 and cycle detection).
- Does the pass correctly handle multi-threaded computation graphs (execution thread filtering)?
- If the pass modifies the graph mid-traversal, does it call `Cleanup()` before dependent passes run?

### 2. GPU Optimization Pass Pipeline
- Is the new pass inserted in the correct stage of `GpuCompiler::RunHloPasses()` / `OptimizeHloPostLayoutAssignment()`?
- Does the pass interact safely with adjacent passes (layout assignment, float normalization, SPMD partitioning)?
- If a pass is wrapped in `HloPassFix<>`, has convergence been verified?
- Are `HloCSE` and `HloDCE` run after the new pass where needed?
- For algebraic rewrites: are all edge cases (zero-sized tensors, scalar shapes, dynamic shapes) handled?

### 3. Fusion & Fission
- If the change touches `PriorityFusion`, `MultiOutputFusion`, or `gpu_fusible.cc`:
  - Is the cost model estimate (`time_unfused - time_fused`) accurate?
  - Does the change respect `FusionFitsInParameterLimit()` and `FusionFitsInBudget()` (max 96 operands)?
  - Are IR size guards (`kMaxIRSize = 10000`, `kMaxBasicBlockSplitsPerFusion = 10`) respected?
  - Is `HloFusionAnalysisCache` properly invalidated after graph mutations?
- If a new `EmitterFusionKind` is added, does it have a corresponding emitter in `xla/backends/gpu/codegen/emitters/`?
- For fission passes (`ReductionSplitter`, `SplitKGemmRewriter`, `VariadicOpSplitter`): are correctness constraints documented and tested?
- Is `FusionProcessDumpProto` updated to log new fusion decisions?

### 4. Triton Integration
- If the change touches `xla/backends/gpu/codegen/triton/`:
  - Does the HLO→XTile→Triton IR lowering handle all relevant dtypes (`xla/backends/gpu/codegen/triton/support.cc`)?
  - Are new `ttxla.*` dialect ops defined in `triton_xla_ops.td` with correct semantics?
  - Do new passes in `/transforms/` handle rank-1 edge cases and TMA constraints (Hopper+)?
  - Is the ROCDL path in `compilation_pipeline_rocm.cc` updated alongside the CUDA path?
  - Are shared memory limits validated against `device_info.shared_memory_per_block_optin()`?
  - For collective fusions: are `ttxla.block_barrier` / `ttxla.atomic_write` semantics correct?
- For `BlockLevelFusionConfig` changes: are all parameters (`num_warps`, `num_ctas`, `num_stages`) validated as > 0?

### 5. Autotuner
- If the change touches `xla/backends/autotuner/` or `xla/service/gpu/autotuning/`:
  - **Cache key**: if the config format changes, is the cache version bumped (currently v24 in `autotune_cache_key.h`)?
  - **Search space**: for Triton config changes, does `TritonDotFusionSearchSpace::GenerateConfigs()` produce valid configs for the affected shapes and hardware?
  - **Correctness checking**: if `CanProduceWrongResults()` changes for any backend, is the relative tolerance adjusted?
  - **ROCm factory**: is `factory_rocm.cc` updated if new backends are added? (Backend order: Triton → MIOpen → rocBLAS → hipBLASLt)
  - **Default configs**: are `default_configs/rocm.txtpb` and CUDA equivalents updated for affected architectures?
  - Does the change work in `READ_WRITE` and `READ` (inference-time) cache modes?

### 6. AMD/ROCm Parity
- Does every CUDA-path change have a corresponding ROCm path update?
  - `compilation_pipeline_cuda.cc` ↔ `compilation_pipeline_rocm.cc`
  - `factory_cuda.cc` ↔ `factory_rocm.cc`
  - `stream_executor/cuda/` ↔ `stream_executor/rocm/`
- If new float types are used: does `convert_float_amd.cc` handle them? (BF16 and F8 semantics differ between vendors.)
- Are ROCm compute capability checks (`gfx90a`, `gfx942`, etc.) consistent with CUDA compute capability checks?
- If RCCL / MIOpen / hipBLASLt APIs are called, are error codes wrapped with `rocm_status.h`?
- Are ROCm-specific kernel files (e.g., `*_rocm.cu.cc`) added where CUDA-specific kernels are added?
- Do Bazel `BUILD` files include ROCm targets where CUDA targets are added?

### 7. Collective Operations
- For changes to all-reduce / all-gather / reduce-scatter / collective-permute:
  - Are NCCL (CUDA) and RCCL (ROCm) paths both updated?
  - Does `CollectivePipeliner` still correctly overlap compute and communication?
  - Are ragged collective variants (`RaggedAllToAllDecomposer`) accounted for?
- For new collective HLO ops: is `GpuHloCostAnalysis::HandleAllReduce()` (or equivalent) implemented?

### 8. Performance Model
- If `GpuHloCostAnalysis` or `GpuPerformanceModel` is modified:
  - Are `BytesTransferred`, FLOPs, and IR size estimates accurate for the new op/fusion?
  - Is `CommonElementwiseUtilization` updated for new elementwise patterns?
  - Does `ProducerConsumerMergedTooLarge()` guard against oversized IR?

### 9. Testing
- Are HLO-level unit tests added (filecheck or C++ tests with `HloTestBase`)?
- Are GPU backend tests added covering both CUDA and ROCm?
- For autotuner changes: are cache hit/miss tests included?
- For new fusion kinds: is an end-to-end correctness test included?
- For Triton changes: is a `TritonFusionNumericsVerifier`-compatible test added?
- Do tests cover edge cases: zero-sized tensors, scalar inputs, dynamic shapes, multi-device?

### 10. General C++ Correctness
- **Ownership & lifetime**: Does ownership transfer use `std::unique_ptr`? Are raw pointers used only for non-owning observation? Are there dangling references from `absl::string_view`, `absl::Span`, or `llvm::StringRef` outliving the data they point to?
- **Move semantics**: Are large objects moved rather than copied when passed by value? Is `std::move` used on the last use of a local? Are moved-from objects not accessed afterward?
- **RAII**: Are resources (locks, streams, allocations) managed with RAII wrappers (`absl::MutexLock`, smart pointers) rather than manual acquire/release?
- **Thread safety**: Are shared mutable fields annotated `ABSL_GUARDED_BY(mu_)`? Are mutexes `mutable` when locked inside `const` methods? Is `absl::MutexLock` used rather than manual lock/unlock?
- **Integer overflow / narrowing**: Are 64→32-bit casts guarded? Are `int64_t` used for sizes and indices consistently (XLA convention)? Are signed/unsigned comparisons avoided?
- **Const correctness**: Are function parameters, local variables, and member functions marked `const` where appropriate?
- **Initialization**: Are class members initialized in declaration order? Are there uninitialized variables on any code path? Are braced initializers (`{}`) used to avoid narrowing?
- **Error path leaks**: If a function acquires resources and then returns early via `TF_RETURN_IF_ERROR` or `TF_ASSIGN_OR_RETURN`, are those resources cleaned up?
- **UB risks**: Are there null dereferences, out-of-bounds accesses, or use-after-free patterns? Is `CHECK` / `DCHECK` used appropriately (`CHECK` for invariants that indicate bugs, `DCHECK` for expensive debug-only assertions)?

### 11. XLA Project-Specific C++ Conventions
- **Error handling**: Are `absl::Status` / `absl::StatusOr<T>` used for fallible operations? Are `TF_RETURN_IF_ERROR()`, `TF_ASSIGN_OR_RETURN()`, and `TF_RET_CHECK()` used correctly (not `tsl::Status` / `tsl::StatusOr`, not `tsl::Status::OK()` — use `OkStatus()` instead)?
- **Prohibited APIs** (enforced by CI `check_contents.yml`):
  - No `tsl::Status` or `tsl::StatusOr` — use unqualified `Status` / `StatusOr`
  - No `tsl::Status::OK()` — use `OkStatus()`
  - No `std::call_once` — use `absl::call_once`
  - No Abseil compatibility shims (`absl::optional`, `absl::any`, `absl::variant`, `absl::make_unique`, `absl::nullopt`) — use `std::` equivalents
  - No TF/TSL legacy types (`gtl::FlatMap`, `gtl::FlatSet`, `gtl::InlinedVector`, `strings::StrCat`, `strings::Printf`, `str_util::*`, `tensorflow::StringPiece`) — use Abseil equivalents
- **Containers**: Prefer `absl::flat_hash_map` / `absl::flat_hash_set` over `std::unordered_*`. Use `absl::btree_map` when ordered iteration is needed. Use `absl::InlinedVector<T, N>` for small vectors.
- **Strings**: Use `absl::string_view` for parameters, `absl::StrCat()` / `absl::StrFormat()` / `absl::StrJoin()` / `absl::StrAppend()` / `absl::Substitute()` for string construction.
- **Logging**: Use `LOG(INFO/WARNING/ERROR)` and `VLOG(level)`. Use `XLA_SCOPED_LOGGING_TIMER()` for performance instrumentation.
- **Assertions**: Use `CHECK()` / `CHECK_EQ()` / `CHECK_NE()` etc. for fatal invariants. Use `DCHECK()` for debug-only assertions. Use `TF_RET_CHECK()` when the caller can handle the error.
- **Map utilities**: Prefer `FindOrDie()`, `FindOrDefault()`, `ContainsKey()`, `InsertOrDie()` from `xla/map_util.h` where appropriate.
- **Namespace**: Code lives in `namespace xla { ... }`. Use anonymous namespaces for file-local helpers. Use namespace aliases for verbose paths (e.g., `namespace se = ::stream_executor;`).
- **Include order** (enforced by `.clang-format`): (1) corresponding header, (2) C/C++ system headers separated by blank line, (3) third-party headers grouped by library: gtest/gmock, absl, llvm, mlir, protobuf, xla, tsl, triton.
- **Formatting**: Google C++ style (`.clang-format` BasedOnStyle: Google). Pointer binds to type (`int* p`, not `int *p`). CI enforces `clang-format` v17 on all `.cc`/`.h` files.
- **Bazel BUILD**: Targets must be minimal and correctly scoped. `buildifier` linting is enforced. ROCm targets must be included alongside CUDA targets.

### 12. Code Organization & Debug Support
- Are large new passes kept under ~500 lines per file? (Reference material in separate files.)
- Are new debug dump points added (HLO dumping, `FusionProcessDumpProto`)?
- If a new proto field is added: is the proto version/compatibility handled?

---

## Output Format

Begin directly with the structured review below. Do not include reasoning, analysis steps, or thinking-out-loud before it.

```
## Summary
<2–3 sentence overview of what the PR does>

## Findings
- [file:line] <description and why it matters>
```
