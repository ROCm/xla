---
layout: distill
permalink: /jax-rocm-stack.html
title: "The JAX/ROCm Stack"
description: "How JAX traces a program into JAXPR and StableHLO, how XLA lowers that onto ROCm backends, and how MaxText flags change the graph before the compiler runs."
date: 2026-09-16

section_number: 2

previous_section_url: "/mi355-architecture"
previous_section_name: "Chapter 1: MI355X architecture and system topology"

next_section_url: "/rocm-profiling"
next_section_name: "Chapter 3: Profiling a Training Step"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "JAX, tracing and JAXPR"
  - name: "XLA and the GPU pipeline"
  - name: "A map of ROCm kernel backends"
    subsections:
      - name: "Dense Matrix Multiplication"
      - name: "Attention"
      - name: "Low-Precision Training"
      - name: "Collectives and Mixture-of-Experts"
      - name: "Activations and normalization"
      - name: "Other XLA ROCm integrations"
      - name: "Extending the stack"
  - name: "Introducing MaxText"
  - name: "Worked example: Following a MaxText flag though the stack"
    subsections:
      - name: "Changes at the HLO level"
---
## JAX, tracing and JAXPR

JAX adopts a largely functional programming model. Computation is expressed as the composition of pure transformations: functions receive inputs and produce outputs without mutating hidden state. While variables may appear to be reassigned at the Python level, JAX transformations operate on immutable values. The absence of mutable state makes program behaviour easier to reason about because the output of a function depends only on its inputs. More broadly, immutability, referential transparency, and function composition are well-established advantages of the functional programming paradigm; interested readers are referred to [Why Functional Programming Matters](https://www.cs.kent.ac.uk/people/staff/dat/miranda/whyfp90.pdf) (Hughes, 1989).

These properties are also valuable from a compiler perspective. Consider a compiler attempting to reorder operations, fuse kernels, eliminate redundant computations, or distribute work across multiple devices. If arbitrary operations could modify shared state, the compiler would need to conservatively preserve execution order. Pure functions instead give the compiler greater freedom to analyze and transform a program while preserving its semantics. In practice, this enables the aggressive whole-program optimizations that form the foundation of modern accelerator software stacks.

JAX builds on this functional model through *program transformations*. Transformations such as automatic differentiation (`grad`), vectorization (`vmap`), and JIT compilation are applied to user-defined functions rather than individual operations. The JAX authors describe JAX as a system for expressing and composing transformations of numerical programs, an idea that underpins much of the framework's design. Readers interested in this may find the official JAX documentation useful: [JAX 101: Expressing Computations](https://docs.jax.dev/en/latest/101/index.html).

A key enabling mechanism behind these transformations is *tracing*. Rather than immediately executing a function, transformations such as `jax.jit()` first trace the operations performed by the function and construct an IR of the computation. During tracing, concrete values are replaced by tracer objects that record the sequence of primitive operations and the flow of data between them. This representation can then be analyzed, transformed, and compiled for efficient execution on accelerator hardware. The official JAX documentation provides a detailed walkthrough of this process in [Tracing](https://docs.jax.dev/en/latest/tracing.html).

The first compiler-level representation exposed to users is **JAXPR** (JAX Expression Representation). JAXPR can be viewed as JAX's frontend IR, capturing a computation as a sequence of primitive operations and explicit data dependencies. It is intentionally simple, hardware-independent, and well-suited to the transformations that JAX performs, including automatic differentiation, vectorization, and JIT compilation. Readers interested in the broader transformation model may find the JAX documentation on [Transformations and Tracing](https://docs.jax.dev/en/latest/101/index.html) useful supplementary reading.

Below JAXPR lies **StableHLO**, a portable compiler IR shared across the OpenXLA ecosystem. Although many JAX primitives map naturally to StableHLO operations, the relationship is not necessarily one-to-one.


```python
@jax.jit
def f(x, y):
    return x * y + 1
```


The traced function is first represented as JAXPR:

```
{ lambda ; a:f32[] b:f32[].
  let
    c = mul a b
    d = add c 1.0
  in (d,) }
```

Which is subsequently lowered into StableHLO:

```
module {
  func.func @main(%arg0: tensor<f32>,
                  %arg1: tensor<f32>)
      -> tensor<f32> {

    %0 = stablehlo.multiply %arg0, %arg1
    %1 = stablehlo.constant dense<1.0>
    %2 = stablehlo.add %0, %1

    return %2
  }
}
```

StableHLO forms the entry point into the XLA compilation pipeline.


## XLA and the GPU pipeline

Take note that going from JAXPR to StableHLO, we transition from JAX-specific representations to the framework-agnostic XLA stack. In other words, HLO is not just a JAX compiler IR, but a shared representation used across multiple frontends including JAX, TensorFlow, and PyTorch integrations.

OpenXLA maintains [documentation of XLA internals](https://openxla.org/xla):

- [Operation Semantics](https://openxla.org/xla/operation_semantics): reference documentation for HLO primitives.
8
- [From HLO to Thunks](https://openxla.org/xla/hlo_to_thunks): overview of the compiler pipeline.
9
- [GPU Architecture Overview](https://openxla.org/xla/gpu_architecture): GPU-specific compilation details, including HLO examples of sharding, fusion, and layout assignment.
10
- [XLA Emitters](https://openxla.org/xla/emitters): details on XLA's internal kernel emitters and code generation infrastructure.

The main points from the ROCm perspective are:

1. The GPU backend runs platform-specific passes. ROCm and CUDA builds can therefore
produce different shardings, fusions, layouts, and library calls from the same
StableHLO program.

![]({{ '/pages/img/pg2/hlo_to_thunks.svg' | relative_url }})

2. XLA lowers an HLO operation through an XLA GPU emitter, Triton, or an integrated
runtime library.

![]({{ '/pages/img/pg2/hlo-3-path-lower.png' | relative_url }})

3. [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) lets a JAX package or user
register an external ROCm implementation. An FFI call appears in HLO as a
`custom_call`, then executes through the registered handler at runtime.

## A map of ROCm kernel backends

The three lowering paths, XLA code generation, [Triton code generation](https://openxla.org/xla/gpu_architecture#compiler-backend-codegen-and-library-selection), and library selection through `custom_call`, describe *how* an HLO operation reaches executable device code. In practice, however, it is often more useful to think in terms of *what operation is being performed*.

A transformer architecture generally consists of a set of operation classes: matrix multiplication, attention, communication, quantization, normalization, activation functions, routing, and numerical primitives. Multiple backend implementations may exist for each of these operations, with the final lowering path determined by shape, datatype, layout, sharding, compiler configuration, and available backend integrations.

![]({{ '/pages/img/pg2/hlo-3-path-lower-detailed.svg' | relative_url }})

The sections below summarize the primary operation classes encountered in modern transformer workloads, together with the ROCm libraries that may implement them.

- [XLA emitters](https://openxla.org/xla/emitters) and [XLA's Triton backend](https://openxla.org/xla/gpu_architecture#compiler-backend-codegen-and-library-selection) generate kernels from HLO.
- [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/), [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/), [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/), [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/), [hipSOLVER](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/), [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/), and [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/) are
  integrated runtime libraries. XLA has compiler rewrites or thunks for supported
  operations.
- [Transformer Engine](https://github.com/ROCm/TransformerEngine), [JAX-AITER](https://github.com/ROCm/jax-aiter), [MORI](https://github.com/ROCm/mori), and similar packages expose JAX-facing
  operations through FFI or custom primitives. They may call lower-level libraries
  or launch their own kernels.
- [AITER](https://github.com/ROCm/aiter), [Composable Kernel](https://github.com/ROCm/composable_kernel), [AOTriton](https://github.com/ROCm/aotriton), and [FlyDSL](https://github.com/ROCm/FlyDSL) kernels can sit behind those
  JAX-facing packages. They are not always visible in HLO.

For example, [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/) provides a portable BLAS interface while rocBLAS supplies the underlying ROCm implementation. hipBLASLt extends this model with support for advanced GEMM features such as custom layouts, grouped GEMMs, low-precision datatypes, and fused epilogues. Likewise, Transformer Engine and JAX-AITER expose JAX-facing APIs while relying on lower-level kernel libraries to perform the actual computation.

### Dense Matrix Multiplication

GEMMs implement attention projections, MLP projections, output projections, and many optimizer operations.

JAX `dot_general` and StableHLO `dot_general` ultimately become HLO `dot` operations. Depending on the problem being solved, XLA may lower these operations through:

- rocBLAS via hipBLAS;
- hipBLASLt;
- Triton GEMMs;
- XLA-generated kernels; or
- custom implementations exposed through Transformer Engine, JAX-AITER, or other FFI integrations.

The selected implementation depends on the matrix dimensions, batching structure, layouts, datatypes, epilogues, workspace requirements, and compiler configuration.

### Attention

The case studies presented later in this book focus on three attention implementations:

- standard JAX operations lowered by XLA.
- fused attention selected by [Transformer Engine](https://github.com/ROCm/TransformerEngine).
- direct [JAX-AITER](https://github.com/ROCm/jax-aiter) attention through FFI.

[Pallas](https://docs.jax.dev/en/latest/pallas/index.html)/Triton-based attention kernels are another possible implementation route.

Although these implementations compute the same attention function, they differ significantly in their tiling strategies, memory movement, supported datatypes, workspace requirements, and backward-pass implementations.

Transformer Engine itself acts as a dispatch layer, selecting an appropriate backend such as Composable Kernel (CK), AITER, or AOTriton depending on the attention configuration and hardware platform.

### Low-Precision Training

Modern training increasingly relies on formats such as FP8, MXFP8, and MXFP4 to increase arithmetic throughput and reduce memory consumption.

These formats involve more than a storage datatype. Practical implementations typically define quantization schemes, scaling strategies, accumulation rules, custom kernels, and gradient behavior.

XLA can perform mixed-precision lowering and insert datatype conversions where appropriate. Frameworks such as Transformer Engine and JAX-AITER build additional runtime infrastructure on top of these compiler capabilities to support end-to-end low-precision training.

### Collectives and Mixture-of-Experts

Sharding introduces first-class HLO collectives such as `all-reduce`, `all-gather`, `reduce-scatter`, and `all-to-all`. On ROCm, these operations are implemented through RCCL.

MoE training introduces additional types of work:

1. Routing, where tokens are assigned to experts.
2. Dispatch and combine, where tokens move between expert owners.
3. Expert execution, which performs the expert GEMMs.

Routing is typically expressed using ordinary JAX operations. Dispatch and combine may involve RCCL, MORI, [DeepEP](https://github.com/deepseek-ai/DeepEP), or similar communication libraries. Expert execution typically relies on dense GEMMs, GroupedGEMMs, or specialized MoE kernels.

MORI focuses on expert communication and token movement, while expert computation continues to rely on existing GEMM backends.

General sparse linear algebra is a separate category. Although MoE models are sparse at the routing level, individual experts still tend to execute dense matrix multiplications.

### Activations and normalization

RMSNorm, LayerNorm, softmax, SiLU, gating, and related operations are built from
pointwise work and reductions. They may become:

- an XLA fusion
- a Triton fusion
- a JAX-AITER FFI kernel
- a Transformer Engine kernel

Unlike GEMMs or collectives, these operations are often fused into surrounding computation in order to reduce memory traffic and kernel launch overhead.

### Other XLA ROCm integrations

XLA's ROCm runtime also wraps libraries used outside the main transformer path:

- [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) for convolutions and
  other deep-learning primitives;
- [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/) and
  [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/latest/) for Fourier
  transforms;
- [hipSOLVER](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/) and
  [rocSOLVER](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/) for
  factorizations and linear systems; and
- [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/) and
  [rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/) for general
  sparse matrix and vector operations.


### Extending the stack

[XLA FFI](https://docs.jax.dev/en/latest/ffi.html) is the extension point used by
packages such as JAX-AITER. A package defines a JAX-facing operation, lowers it to an
HLO custom call, and registers a ROCm handler that receives the GPU stream and device
buffers. The handler can launch its own kernel or call another ROCm library.

This route gives the extension control over its operation contract and runtime
implementation. It also gives the extension responsibility for abstract evaluation,
batching, autodiff, sharding, layouts, aliases, workspace, and error handling. A
forward handler alone is insufficient for training.

## Introducing MaxText

[MaxText](https://github.com/AI-Hypercomputer/maxtext) is an open-source large language model training framework developed by Google and built on top of JAX. It serves as both a production-oriented training framework and a reference implementation for modern transformer architectures, distributed training techniques, and accelerator optimizations. In practice, MaxText demonstrates how large-scale foundation models can be implemented within the JAX and OpenXLA ecosystem.

MaxText operates one layer above JAX. Rather than directly modifying the XLA compilation pipeline, MaxText influences the JAX program that enters the compiler stack.

At a high level, MaxText code follows one of two paths:

![]({{ '/pages/img/pg2/maxtext-to-rocm-backends.svg' | relative_url }})

Standard MaxText layers lower through ordinary JAX primitives. Specialized integrations such as Transformer Engine and JAX-AITER instead lower through FFI-backed custom JAX primitives, which appear in XLA as `custom_call` operations.

MaxText exposes many of these choices through configuration flags. While such flags may appear as runtime configuration, they frequently alter the generated JAX program itself. Enabling a feature may change which modules are instantiated, which operations are emitted, or whether specialized implementations are selected. Consequently, the effect of a MaxText flag is typically visible before compilation even reaches XLA. The flag modifies the JAX source graph, which in turn changes the generated JAXPR, StableHLO, and subsequent lowering behaviour.



## Worked example: Following a MaxText flag though the stack

An instructive example is changing MaxText's attention implementation from standard
JAX attention to Transformer Engine fused attention:

```yaml
# Standard JAX attention
attention: dot_product

# Transformer Engine fused attention
attention: cudnn_flash_te
```

The `cudnn` prefix is a historical compatibility name. On ROCm, this option
enters Transformer Engine's JAX attention path rather than loading cuDNN.

MaxText parses `attention` as part of its typed configuration, passes it through
the model configuration, and uses it to select the attention implementation.
These links refer to
[`ROCm/maxtext` at `b437942a`](https://github.com/ROCm/maxtext/tree/b437942a5f33704f8438deb948488ad08164285c):

1. [`types.py`](https://github.com/ROCm/maxtext/blob/b437942a5f33704f8438deb948488ad08164285c/src/maxtext/configs/types.py#L619-L625)
   defines the `attention` field.
2. [`llama2.py`](https://github.com/ROCm/maxtext/blob/b437942a5f33704f8438deb948488ad08164285c/src/maxtext/models/llama2.py#L82-L100)
   passes the resolved value into the attention module.
3. [`attention_op.py`](https://github.com/ROCm/maxtext/blob/b437942a5f33704f8438deb948488ad08164285c/src/maxtext/layers/attention_op.py#L1023-L1197)
   contains the implementation dispatch logic.

```python
self.self_attention = Attention(
    config=config,
    attention_kernel=config.attention,
)
```

This configuration determines which
Python code path JAX traces. The dispatch occurs inside
`AttentionOp.apply_attention`:

```python
if self.attention_kernel == "dot_product":
    return self.apply_attention_dot(...)

elif self.attention_kernel == "cudnn_flash_te":
    return self.cudnn_flash_attention(...)
```

With `attention=dot_product`, MaxText remains within standard JAX operations.
The implementation constructs the attention algorithm by computing QK scores,
applying masks, calculating the softmax terms, and multiplying by V:

```python
attn_weights = self.qk_product(query, key, ...)
attn_weights = apply_mask_to_logits(attn_weights, attn_mask)

local_max = jnp.max(logits, axis=-1, keepdims=True)
local_exps_combined = jnp.exp(logits - local_max)
local_sum = jnp.sum(local_exps_combined, axis=-1, keepdims=True)

local_exps = local_exps_combined[..., :s]
local_out = self.wv_product(local_exps, value, ...)
```

Each operation is independently visible to JAX tracing and therefore appears in
the generated HLO at this stage.

With `attention=cudnn_flash_te`, MaxText instead constructs a Transformer Engine
attention module:

```python
from transformer_engine.jax.flax.transformer import DotProductAttention

dpa_layer = DotProductAttention(
    head_dim=head_dim,
    num_attention_heads=self.num_query_heads,
    num_gqa_groups=self.num_kv_heads,
    attn_mask_type=mask_type,
    qkv_layout=qkv_layout,
    # ...
)

return dpa_layer(
    query,
    key,
    value,
    sequence_descriptor=attn_mask,
)
```

JAX now traces a Transformer Engine operation, which lowers through an FFI-backed
custom call rather than exposing the internal attention algorithm. MaxText also
enters Transformer Engine's mesh context, passing the data, tensor, FSDP, and
context-parallel axes so the attention implementation observes the same device
mesh as the rest of the model.

### Changes at the HLO level

The examples below were captured from matched BF16 attention fixtures with Q, K,
and V tensors shaped `[1, 128, 8, 64]`. One fixture uses standard JAX attention,
and the other uses the same Transformer Engine path selected by MaxText.

With standard JAX attention, the HLO exposes the attention algorithm directly.
The first `dot` computes QK scores, the reductions and exponential implement the
softmax, and the final `dot` multiplies the resulting probabilities by V:

[![XLA-rendered HLO subgraph for standard JAX attention]({{ '/pages/img/pg2/hlo-attention-xla.svg' | relative_url }})]({{ '/pages/img/pg2/hlo-attention-xla.svg' | relative_url }})

*Representative subgraph pruned from XLA's literal `before_optimizations` DOT
graph.*

With Transformer Engine, the same pipeline stage contains the external
call. Q, K, V and attention metadata enter
`custom_call_target="te_fused_attn_forward_ffi"`, and a
`get-tuple-element` extracts the forward result.

[![XLA-rendered HLO subgraph for Transformer Engine fused attention]({{ '/pages/img/pg2/hlo-attention-te.svg' | relative_url }})]({{ '/pages/img/pg2/hlo-attention-te.svg' | relative_url }})

*Representative subgraph pruned from XLA's literal `before_optimizations` DOT
graph. Q, K, V, and metadata converge on the
fused-attention custom call.*

This is the central pattern behind many MaxText configuration options. The flag
does not modify XLA directly, rather it changes the JAX program being traced, producing
a different JAXPR, different StableHLO, and a different XLA lowering path.

<h3 markdown=1 class="next-section">Next: [profiling a training step]({{ '/rocm-profiling' | relative_url }}).</h3>

