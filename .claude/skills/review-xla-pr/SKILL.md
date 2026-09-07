---
name: review-xla-pr
description: Review an XLA pull request with deep expertise in HLO optimizations, GPU backend, Triton codegen, autotuner, and AMD/ROCm parity. Use when asked to review an XLA PR or check an XLA change.
argument-hint: [PR-number]
context: fork
agent: general-purpose
allowed-tools: Bash(gh pr view *), Bash(gh pr diff *), Bash(git log *), Bash(git show *), Bash(git diff *), Bash(git blame *), Bash(head *), Bash(grep *), Read, Grep, Glob
---

# XLA PR Review

## IMPORTANT: Do NOT post comments

This skill is read-only. Do NOT post any comments, reviews, or reactions to GitHub.
Do NOT use `gh pr comment`, `gh pr review`, `gh api` with POST/PUT/PATCH/DELETE,
or any other write operation. Only return your findings as text output — the caller
is responsible for posting them.

## Fetching Pull Request Context

Before reviewing, fetch the PR context by running these commands:
- `gh pr view $ARGUMENTS` — PR title, description, author
- `gh pr diff $ARGUMENTS --name-only` — list of changed files
- `gh pr diff $ARGUMENTS` — full diff
- `gh pr view $ARGUMENTS --comments 2>/dev/null || echo "(no comments yet)"` — existing review comments

Then read `xla/AGENTS.md` from the checkout. It is the authoritative upstream coding-convention document (error handling and status macros, `TF_RET_CHECK` vs `DCHECK`, `Decision` instead of `bool`, `auto` in headers, namespaces, MLIR op creation, BUILD rule choices, test bases and test hygiene, phase ordering, performance sensitivity). Treat it as part of this checklist and flag violations of it. The checklist below deliberately does not repeat it; it covers only what `xla/AGENTS.md` leaves out.

If `xla/AGENTS.md` is missing, the PR targets a branch that predates it. Those conventions still apply, so review against the [upstream copy](https://github.com/openxla/xla/blob/main/xla/AGENTS.md) and say in the summary that you could not read it from the checkout.

## Your Task

Review the PR above thoroughly using the checklist below. Be specific: cite file paths and line numbers from the diff. Describe each finding clearly and explain why it matters. Do NOT assign severity labels (no BLOCKER/WARNING/NIT) — let the human reviewer judge importance.

**CRITICAL SCOPE RULE**: Only flag issues that exist in the PR diff itself — lines added or modified by this PR. Do NOT flag pre-existing code that the PR did not touch, even if that code is in the same files. If a pre-existing problem is worth noting, mention it briefly in a separate "Pre-existing issues (out of scope)" section at the end, never as a BLOCKER or WARNING against this PR.

---

## XLA Review Checklist

### 1. HLO IR Correctness
- Does the change preserve the `HloModule` / `HloComputation` / `HloInstruction` invariants?
- If new opcodes or shape semantics are introduced, are all `DfsHloVisitor` `Handle*` methods updated?
- Are `HloPassPipeline` / `HloPassFix` used correctly? Fixed-point passes must not loop infinitely (`HloPassFix::kDefaultIterationLimit` = 25 in `xla/hlo/pass/hlo_pass_fix.h`; a custom limit goes through `HloPassFix::Create(iteration_limit, ...)`).
- Does the pass correctly handle multi-threaded computation graphs (execution thread filtering)?
- If the pass modifies the graph mid-traversal, does it call `Cleanup()` before dependent passes run?

### 2. GPU Optimization Pass Pipeline
- Is the new pass inserted in the correct stage of `GpuCompiler::RunHloPasses()` / `OptimizeHloPostLayoutAssignment()`?
- Does the pass interact safely with adjacent passes (layout assignment, float normalization, SPMD partitioning)?
- If a pass is wrapped in `HloPassFix<>`, has convergence been verified?
- Are `HloCSE` and `HloDCE` run after the new pass where needed?
- For algebraic rewrites: are all edge cases (zero-sized tensors, scalar shapes, dynamic shapes) handled?

### 3. Fusion & Fission
- If the change touches `PriorityFusion` / `MultiOutputFusion` (`xla/backends/gpu/transforms/`) or `xla/service/gpu/gpu_fusible.cc`:
  - Is the cost model estimate (`time_unfused - time_fused`) accurate?
  - Does the change respect `FusionFitsInParameterLimit()` and `FusionFitsInBudget()`? The cap is `MaxOperandsAndOutputsPerFusion()` = 96 in `xla/service/gpu/gpu_fusible.h`.
  - Are IR size guards respected (`GpuHloCostAnalysis::kMaxIRSize` = 10000, `kMaxBasicBlockSplitsPerFusion` = 10, both in `xla/service/gpu/model/gpu_hlo_cost_analysis.h`)?
  - Is `HloFusionAnalysisCache` properly invalidated after graph mutations?
- If a new `EmitterFusionKind` is added (enum in `xla/service/gpu/hlo_fusion_analysis.h`), is it dispatched in `xla/backends/gpu/codegen/fusions.cc` with a corresponding emitter under `xla/backends/gpu/codegen/emitters/`?
- For fission passes in `xla/backends/gpu/transforms/` (`ReductionSplitter`, `SplitkRewriter`, `VariadicOpSplitter`): are correctness constraints documented and tested?
- Is `FusionProcessDumpProto` updated to log new fusion decisions?

### 4. Triton Integration
- If the change touches `xla/backends/gpu/codegen/triton/`:
  - Does the HLO→XTile→Triton IR lowering handle all relevant dtypes (`xla/backends/gpu/codegen/triton/support.cc`)?
  - Are new `triton_xla.*` dialect ops defined in `xla/backends/gpu/codegen/triton/ir/triton_xla_ops.td` with correct semantics? (The dialect mnemonic is `triton_xla`; `TTXLA_` is only the TableGen class prefix.)
  - Do new passes in `xla/backends/gpu/codegen/triton/transforms/` handle rank-1 edge cases and TMA constraints (Hopper+)?
  - Is the ROCDL path in `compilation_pipeline_rocm.cc` updated alongside `compilation_pipeline_cuda.cc`?
  - Are shared memory limits validated against `device_info.shared_memory_per_block_optin()`?
  - For collective fusions: are `triton_xla.block_barrier` / `triton_xla.atomic_write` / `triton_xla.atomic_spin_wait` semantics correct?
- For `BlockLevelFusionConfig` changes: are all parameters (`num_warps`, `num_ctas`, `num_stages`) validated as > 0?

### 5. Autotuner
- The autotuner lives in three places: backend-agnostic core in `xla/backends/autotuner/`, GPU codegen backends in `xla/backends/gpu/autotuner/`, and the legacy cache/key code in `xla/service/gpu/autotuning/`. If the change touches any of them:
  - **Cache key**: if the config format changes, is `AutotuneCacheKey::kCurrentVersion` bumped in `xla/service/gpu/autotuning/autotune_cache_key.h` (currently 51), with the accompanying comment updated to say why?
  - **Cache store**: changes to `AutotunerCacheInterface` implementations (`directory_store`, `in_memory_store`, `tiered_cache`) must honour every `CacheMode` (`kReadOnly`, `kReadAppend`, `kReadWrite`, `kWriteOnly`) and the `kLoose` / exact match modes in `xla/backends/autotuner/autotuner_cache_interface.h`.
  - **Search space**: for Triton config changes, does `TritonDotFusionSearchSpace::GenerateConfigs()` (`xla/backends/gpu/autotuner/triton/dot_search_space.cc`) produce valid configs for the affected shapes and hardware?
  - **Correctness checking**: `CodegenBackend::CanProduceWrongResults()` (`xla/backends/autotuner/codegen_backend.h`) decides whether a config is buffer-compared. If it changes for any backend, is the relative tolerance adjusted?
  - **ROCm factory**: is `xla/backends/gpu/autotuner/factory_rocm.cc` updated if new backends are added? Current registration order in `GetCodegenBackendsForROCm`: Triton → MIOpen → hipBLASLt → Fission → NativeEmitter → BlockLevelEmitter. Compare against `factory_cuda.cc`.
  - **Default configs**: are `xla/backends/gpu/autotuner/triton/default_configs/{rocm,mi300,mi350}.txtpb` and the CUDA equivalents updated for affected architectures?

### 6. AMD/ROCm Parity
- Does every CUDA-path change have a corresponding ROCm path update?
  - `xla/backends/gpu/codegen/triton/compilation_pipeline_{cuda,rocm}.cc`
  - `xla/backends/gpu/autotuner/factory_{cuda,rocm}.cc`
  - `xla/stream_executor/cuda/` ↔ `xla/stream_executor/rocm/`
- If new float types are used: does `xla/backends/gpu/codegen/emitters/transforms/convert_float_amd.cc` handle them? (BF16 and F8 semantics differ between vendors.)
- Are ROCm compute capability checks (`gfx90a`, `gfx942`, etc.) consistent with CUDA compute capability checks?
- If RCCL / MIOpen / hipBLASLt APIs are called, are error codes wrapped with `xla/stream_executor/rocm/rocm_status.h`?
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
(Test base classes, anonymous namespaces, `xla_cc_test`, and flake-freedom are covered by `xla/AGENTS.md`.)

- Are HLO-level unit tests added (FileCheck, or C++ tests)?
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
- **Error path leaks**: If a function acquires resources and then returns early via `ABSL_RETURN_IF_ERROR`, `ABSL_ASSIGN_OR_RETURN`, or `TF_RET_CHECK`, are those resources cleaned up?
- **UB risks**: Are there null dereferences, out-of-bounds accesses, or use-after-free patterns? (For which assertion macro to use, follow `xla/AGENTS.md`.)

### 11. XLA Conventions Not Covered by `xla/AGENTS.md`

Error handling and status macros, `TF_RET_CHECK` vs `DCHECK`, `Decision`, `auto` in headers, flat namespaces, MLIR `OpTy::create`, `xla_cc_test` / `tf_proto_library`, and hot-path allocation cost all come from `xla/AGENTS.md`. Do not re-derive them here. The items below are the ones it does not state.

- **Deprecation status of the `TF_*` macros**, which `xla/AGENTS.md` does not spell out:
  - `TF_RETURN_IF_ERROR` and `TF_ASSIGN_OR_RETURN` are CI-blocked by the "Check for TF Status Macros" step in `check_contents.yml`. `xla/tsl/`, `xla/python/`, and `xla/pjrt/` are excluded from that check, so new uses under those paths slip past CI and need to be caught in review.
  - `TF_ASSERT_OK_AND_ASSIGN` / `TF_ASSERT_OK` / `TF_EXPECT_OK` are marked `ABSL_DEPRECATED` but are *not* CI-blocked, so new uses still compile and need to be caught in review.
  - `tsl::errors::InvalidArgument()` and the rest of the `tsl::errors::*` factory family are deprecated → `absl::InvalidArgumentError(absl::StrCat(...))` and equivalents.
- **Prohibited APIs** (enforced by CI `check_contents.yml`):
  - No `tsl::Status` or `tsl::StatusOr` — use unqualified `Status` / `StatusOr`
  - No `tsl::Status::OK()` — use `OkStatus()`
  - No `std::call_once` — use `absl::call_once`
  - No Abseil compatibility shims: `absl::any`, `absl::get`, `absl::get_if`, `absl::make_any`, `absl::make_unique`, `absl::make_optional`, `absl::nullopt`, `absl::nullopt_t`, `absl::optional`, `absl::underlying_type_t`, `absl::variant`, `absl::visit` — use the `std::` equivalents
  - No TF/TSL legacy types: `gtl::FlatMap`, `gtl::FlatSet`, `gtl::InlinedVector`, `gtl::optional`, `strings::StrCat`, `strings::StrAppend`, `strings::Printf`, `strings::Appendf`, `strings::safe_strto64`, `strings::safe_strtof`, `str_util::*`, `tensorflow::StringPiece` — use Abseil equivalents
  - Header guards are checked by `build_tools/lint/check_header_guards.py`
  - Python files: no bare `print()` (suppress with `DISABLE_DEBUG_PRINT_CHECK`), and use `mock.patch.object` rather than `mock.patch(`
- **Containers**: Prefer `absl::flat_hash_map` / `absl::flat_hash_set` over `std::unordered_*`. Use `absl::btree_map` when ordered iteration is needed. Use `absl::InlinedVector<T, N>` for small vectors.
- **Strings**: Use `absl::string_view` for parameters, and `absl::StrCat()` / `absl::StrFormat()` / `absl::StrJoin()` / `absl::StrAppend()` / `absl::Substitute()` for string construction.
- **Logging**: Use `LOG(INFO/WARNING/ERROR)` and `VLOG(level)`. Use `XLA_SCOPED_LOGGING_TIMER()` for performance instrumentation.
- **Assertions**: Use `CHECK()` / `CHECK_EQ()` / `CHECK_NE()` etc. for fatal invariants that indicate a bug with no recoverable caller.
- **Map utilities**: Prefer `FindOrDie()`, `FindOrDefault()`, `ContainsKey()`, `InsertOrDie()` from `xla/map_util.h` where appropriate.
- **Namespace**: Use anonymous namespaces for file-local helpers. Use namespace aliases for verbose paths (e.g., `namespace se = ::stream_executor;`).
- **Include order** (enforced by `.clang-format`): (1) corresponding header, (2) C/C++ system headers separated by blank line, (3) third-party headers grouped by library: gtest/gmock, absl, llvm, mlir, protobuf, xla, tsl, triton.
- **Formatting**: Pointer binds to type (`int* p`, not `int *p`). CI runs `build_tools/ci/run_clang_format.sh`, pinned to clang-format 17.0.6, over the diff against `main`.
- **Bazel BUILD**: Targets must be minimal and correctly scoped. `buildifier` and the DWYU (depend-on-what-you-use) check are enforced. ROCm targets must be included alongside CUDA targets.

### 12. Code Organization & Debug Support
- Are large new passes kept under ~500 lines per file? (Reference material in separate files.)
- Are new debug dump points added (HLO dumping, `FusionProcessDumpProto`)?
- If a new proto field is added: is the proto version/compatibility handled?

### 13. PR Size
- Count the total delta from the diff already in context: lines beginning with `+` (excluding `+++` headers) as additions, lines beginning with `-` (excluding `---` headers) as deletions. Total delta is additions + deletions across **all** changed files. The [upstream size check](https://github.com/openxla/xla/blob/main/.github/workflows/pr_size_check.py) uses GitHub's own `pull_request.additions` / `.deletions` and does **not** exclude test files, so do not exclude them either. Only the highest matching threshold is reported:
  - Total delta >= 1000: "🔴 This PR has a very large delta of over 1000. In order to enable an effective code review, please break the PR down into smaller and more focused PRs."
  - Total delta >= 500: "⚠️ This PR has a large delta of over 500. Consider breaking the PR down into smaller PRs for a faster code review."

---

## Output Format

Begin directly with the structured review below. Do not include reasoning, analysis steps, or thinking-out-loud before it.

```
## Summary
<2–3 sentence overview of what the PR does>

## Findings
- [file:line] <description and why it matters>
```
