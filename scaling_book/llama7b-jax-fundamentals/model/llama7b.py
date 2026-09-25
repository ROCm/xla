# Copyright 2023 Meta AI, EleutherAI and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training-only Flax Llama 2 model.

The architecture core is adapted from Hugging Face Transformers v4.53.2,
commit ``37f8b0b53512e6aae0cfd15746c133c101783178``:
https://github.com/huggingface/transformers/blob/37f8b0b53512e6aae0cfd15746c133c101783178/src/transformers/models/llama/modeling_flax_llama.py

Changes from that source are intentionally limited to this experiment's needs:

* replace ``LlamaConfig`` and pretrained-model machinery with ``ModelConfig``;
* remove decoding caches, padding masks, dropout and model-output wrappers;
* retain scanned layers and named rematerialization checkpoints;
* dispatch attention explicitly to XLA, Transformer Engine, JAX-AITER or Triton;
* preserve MaxText-compatible ``DenseGeneral`` projection layouts;
* retain separate compute and parameter dtypes plus the MaxText FLOP accounting.

FAITHFULNESS

Shapes, dtypes and the FLOP count are matched to MaxText's llama2-7b deliberately, so the
comparison is like for like:

  32 layers, d_model 4096, 32 query heads, 32 KV heads (MHA, not GQA), head_dim 128,
  SwiGLU MLP of 11008, vocab 32000, RMSNorm eps 1e-5, RoPE theta 10000, untied output
  projection. 6,738,415,616 parameters.

Verified against the MaxText run's dumped HLO entry layout, which is the ground truth for
what actually got compiled.

THE ONE PLACE THIS CANNOT MATCH MaxText, AND IT MATTERS

Attention. MaxText ran `attention: cudnn_flash_te`, which reaches Transformer Engine's
fused kernel and lands on AITER: the kernel trace shows `aiter::fmha_fwd_hd128_bf16_causal`
and `aiter::fmha_bwd_hd128_bf16_causal_a32_psskddv`.

`jax.nn.dot_product_attention` does NOT get there. Measured in this container at these
shapes: `implementation="cudnn"` raises "cuDNN is not detected" on ROCm, and the XLA path
emits no custom call for attention at all -- profiling it shows XLA's own `fusion_1` and
`gemm_fusion_dot_*`, and zero AITER kernels. So in raw JAX on ROCm you get XLA-generated
attention, and the vendor's fused kernel is reachable only through Transformer Engine.

The attention stage compares four explicit raw-JAX call paths:

  xla   jax.nn.dot_product_attention. What raw JAX gives you. Materialises the score
        matrix, so memory grows with sequence squared.
  te    transformer_engine.jax.attention.fused_attn, called directly. The same AITER
        kernel MaxText ended up on, without MaxText -- TE is a kernel library here, not a
        framework.
  aiter jax_aiter.mha.flash_attn_func, called directly through JAX FFI.
  triton Tokamax Pallas-Triton flash attention.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import subprocess
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Everything the architecture needs. Populated from configs/jax.yml."""

    emb_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    head_dim: int = 128
    mlp_dim: int = 11008
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    rope_min_timescale: int = 1
    rope_max_timescale: int = 10_000

    # bf16 compute against an fp32 master copy. `dtype` is what the GEMMs run in;
    # `weight_dtype` is what the optimizer keeps and updates. See train.py.
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.float32

    attention: str = "xla"
    remat_policy: str = "minimal_with_context"
    scan_layers: bool = True

    @property
    def n_params(self) -> int:
        """Closed form, so the config can be checked without building the model."""
        attn = 2 * self.emb_dim * self.n_heads * self.head_dim + 2 * self.emb_dim * self.n_kv_heads * self.head_dim
        mlp = 3 * self.emb_dim * self.mlp_dim
        per_layer = attn + mlp + 2 * self.emb_dim  # two RMSNorm scales
        return per_layer * self.n_layers + 2 * self.vocab_size * self.emb_dim + self.emb_dim


# --------------------------------------------------------------------------- pieces


class LlamaRMSNorm(nn.Module):
    """Hugging Face's Llama RMSNorm with explicit parameter dtype."""

    dim: int
    eps: float = 1e-5
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.weight_dtype)
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2).mean(-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return scale.astype(self.dtype) * jnp.asarray(hidden_states, dtype=self.dtype)


def create_sinusoidal_positions(
    position_ids: jax.Array,
    dim: int,
    min_timescale: int,
    max_timescale: int,
) -> tuple[jax.Array, jax.Array]:
    """Adapt Hugging Face's Llama sinusoidal table to dynamic sequence lengths."""
    fraction = jnp.arange(0, dim, 2) / dim
    inv_freq = 1.0 / (min_timescale * (max_timescale / min_timescale) ** fraction)
    freqs = position_ids[..., None] * inv_freq
    embedding = jnp.concatenate((freqs, freqs), axis=-1)[:, :, None, :]
    return jnp.sin(embedding), jnp.cos(embedding)


def rotate_half(tensor: jax.Array) -> jax.Array:
    """Rotate half the head dimension, following Hugging Face Llama."""
    half = tensor.shape[-1] // 2
    return jnp.concatenate((-tensor[..., half:], tensor[..., :half]), axis=-1)


def apply_rotary_pos_emb(
    tensor: jax.Array,
    sin_pos: jax.Array,
    cos_pos: jax.Array,
) -> jax.Array:
    """Apply Hugging Face's rotate-half RoPE convention."""
    return tensor * cos_pos.astype(tensor.dtype) + rotate_half(tensor) * sin_pos.astype(tensor.dtype)


class LlamaAttention(nn.Module):
    """Hugging Face Llama projections and RoPE with selectable attention kernels."""

    cfg: ModelConfig

    def attend(self, q, k, v):
        """Causal attention over [B, S, N, H].

        Scale is 1/sqrt(head_dim) in both, applied by the attention rather than folded
        into the query projection, so the arms are numerically comparable.
        """
        cfg = self.cfg
        scale = 1.0 / (cfg.head_dim**0.5)

        if cfg.attention == "xla":
            # A batched dot, a causal mask, a softmax and a second dot, which XLA fuses as
            # it sees fit. Nothing vendor-specific is reachable from here: profiled at
            # these shapes it runs XLA's own fusion_1 and gemm_fusion_dot_*, and the score
            # tensor [B, N, S, S] is 8 GiB at batch 4, 32 heads, sequence 4096. That
            # number is the entire reason flash attention exists.
            return jax.nn.dot_product_attention(q, k, v, scale=scale, is_causal=True, implementation="xla")

        if cfg.attention == "te":
            # Transformer Engine's fused attention as a plain Flax submodule. This is the
            # identical object MaxText constructs -- MaxText only adds an nnx wrapper
            # around it -- so this arm reproduces the archived runs' AITER kernels without
            # any of the framework. Imported lazily so the `xla` arm never touches TE.
            if os.environ.get("ROCPROF_SKIP_TE_ROCM_PRELOAD") == "1":
                # rocprof puts the complete ROCm SDK directory on
                # LD_LIBRARY_PATH. TE's wheel initializer then preloads a
                # second absolute copy of those libraries, causing duplicate
                # LLVM option registration during PMC collection.
                import rocm_sdk

                initialize_process = rocm_sdk.initialize_process
                subprocess_run = subprocess.run

                def profiler_safe_run(command, *args, **kwargs):
                    if any("is_fp8_fnuz" in str(part) for part in command):
                        return subprocess.CompletedProcess(command, 1)
                    return subprocess_run(command, *args, **kwargs)

                rocm_sdk.initialize_process = lambda **kwargs: None
                subprocess.run = profiler_safe_run
                try:
                    from transformer_engine.jax.flax import DotProductAttention
                finally:
                    rocm_sdk.initialize_process = initialize_process
                    subprocess.run = subprocess_run
            else:
                from transformer_engine.jax.flax import DotProductAttention

            return DotProductAttention(
                head_dim=cfg.head_dim,
                num_attention_heads=cfg.n_heads,
                num_gqa_groups=cfg.n_kv_heads,
                attn_mask_type="causal",
                attn_bias_type="no_bias",
                attention_dropout=0.0,
                # No `dtype=`: TE deprecated it and takes the dtype from the inputs, which
                # are already bf16. It also carries no parameters, so both arms give a
                # parameter-identical model and differ only in which kernel runs.
                qkv_layout="BSHD_BSHD_BSHD",
                scale_factor=scale,
                transpose_batch_sequence=False,
                max_segments_per_seq=1,
                name="te_attention",
            )(q, k, v, deterministic=True)

        if cfg.attention == "aiter":
            from jax_aiter.mha import flash_attn_func

            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
                deterministic=True,
            )[0]

        if cfg.attention == "triton":
            import tokamax

            # Tokamax otherwise infers this from `device.compute_capability`.
            # ROCm reports that value as "gfx950", while Tokamax 0.0.12-0.0.14
            # assumes an NVIDIA-style float such as "9.0". The dot algorithm is
            # already determined by the input dtype, so make it explicit.
            precision = {
                jnp.dtype(jnp.bfloat16): jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                jnp.dtype(jnp.float16): jax.lax.DotAlgorithmPreset.F16_F16_F32,
                jnp.dtype(jnp.float32): jax.lax.DotAlgorithmPreset.F32_F32_F32,
            }[q.dtype]

            from tokamax._src import gpu_utils

            compute_capability = str(getattr(jax.devices()[0], "compute_capability", ""))
            if compute_capability.startswith("gfx"):
                # The Triton heuristic only asks whether this is NVIDIA sm80.
                # `None` correctly selects its architecture-neutral default.
                gpu_utils._compute_capability = lambda device=None: None
                # Both the forward op and its deferred VJP run this guard.
                gpu_utils.has_triton_support = lambda device=None: True

            # Tokamax's hardware guard currently recognizes NVIDIA compute
            # capabilities only; JAX 0.11's ROCm Pallas backend supports gfx950.
            return tokamax.dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                is_causal=True,
                precision=precision,
                implementation="triton",
            )

        raise ValueError(
            f"unknown attention implementation: {cfg.attention!r}; "
            "expected xla, te, aiter, or triton"
        )

    @nn.compact
    def __call__(self, hidden_states, position_ids):
        cfg = self.cfg
        dense = functools.partial(
            nn.DenseGeneral,
            axis=-1,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
        )
        query = dense(features=(cfg.n_heads, cfg.head_dim), name="q_proj")(hidden_states)
        key = dense(features=(cfg.n_kv_heads, cfg.head_dim), name="k_proj")(hidden_states)
        value = dense(features=(cfg.n_kv_heads, cfg.head_dim), name="v_proj")(hidden_states)

        # These names are what the remat policy in remat_policy() matches on, so a tensor
        # is checkpointed or recomputed by being named here. Renaming one silently changes
        # the memory profile.
        query = checkpoint_name(query, "query_proj")
        key = checkpoint_name(key, "key_proj")
        value = checkpoint_name(value, "value_proj")

        sin_pos, cos_pos = create_sinusoidal_positions(
            position_ids,
            cfg.head_dim,
            cfg.rope_min_timescale,
            cfg.rope_max_timescale,
        )
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        context = checkpoint_name(self.attend(query, key, value), "context")
        output = nn.DenseGeneral(
            features=cfg.emb_dim,
            axis=(-2, -1),
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
            name="o_proj",
        )(context)
        return checkpoint_name(output, "out_proj")


class LlamaMLP(nn.Module):
    """Hugging Face Llama's SwiGLU MLP."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, hidden_states):
        cfg = self.cfg
        dense = functools.partial(
            nn.Dense,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
        )
        gate = checkpoint_name(dense(cfg.mlp_dim, name="gate_proj")(hidden_states), "mlpwi_0")
        up = checkpoint_name(dense(cfg.mlp_dim, name="up_proj")(hidden_states), "mlpwi_1")
        hidden_states = jax.nn.silu(gate) * up
        return checkpoint_name(dense(cfg.emb_dim, name="down_proj")(hidden_states), "mlpwo")


class LlamaDecoderLayer(nn.Module):
    """Training-only adaptation of Hugging Face's FlaxLlamaDecoderLayer."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, hidden_states, position_ids):
        cfg = self.cfg
        residual = hidden_states
        hidden_states = LlamaRMSNorm(
            cfg.emb_dim,
            cfg.norm_eps,
            cfg.dtype,
            cfg.weight_dtype,
            name="input_layernorm",
        )(hidden_states)
        hidden_states = residual + LlamaAttention(cfg, name="self_attn")(hidden_states, position_ids)

        residual = hidden_states
        hidden_states = LlamaRMSNorm(
            cfg.emb_dim,
            cfg.norm_eps,
            cfg.dtype,
            cfg.weight_dtype,
            name="post_attention_layernorm",
        )(hidden_states)
        hidden_states = residual + LlamaMLP(cfg, name="mlp")(hidden_states)

        # (carry, y) because nn.scan requires it. There is no per-layer output to stack,
        # so y is None and the residual stream is the whole carry.
        return hidden_states, None


def remat_policy(name: str):
    """Which intermediates survive the forward pass, by checkpoint_name.

    This is the memory/compute dial, and the names are the ones assigned above. The
    default mirrors MaxText's `minimal_with_context` exactly so the archived runs are a
    fair reference: everything expensive is kept, and only cheap elementwise work is
    recomputed in the backward pass. It is the fastest and hungriest end of the trade-off.

    `full` is the opposite end -- save nothing, recompute the whole layer -- and is the
    setting to reach for when a longer sequence stops fitting.
    """
    if name == "full":
        return None  # jax.checkpoint's default: keep nothing but the layer input
    if name == "minimal_with_context":
        return jax.checkpoint_policies.save_only_these_names(
            "query_proj", "key_proj", "value_proj", "context", "out_proj", "mlpwi_0", "mlpwi_1", "mlpwo"
        )
    if name == "save_qkv_proj":
        return jax.checkpoint_policies.save_only_these_names("query_proj", "key_proj", "value_proj")
    if name == "none":
        return jax.checkpoint_policies.everything_saveable
    raise ValueError(f"unknown remat policy: {name!r}")


class Llama(nn.Module):
    """Training-only adaptation of Hugging Face's FlaxLlamaForCausalLM."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, position_ids):
        cfg = self.cfg
        hidden_states = nn.Embed(
            cfg.vocab_size,
            cfg.emb_dim,
            embedding_init=nn.initializers.normal(stddev=0.01),
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            name="embed_tokens",
        )(input_ids.astype(jnp.int32))

        layer = nn.remat(
            LlamaDecoderLayer,
            prevent_cse=not cfg.scan_layers,
            policy=remat_policy(cfg.remat_policy),
        )
        if cfg.scan_layers:
            # The 32 layers become one jax.lax.scan, so the compiler sees a loop over one
            # body instead of 32 near-identical copies. Two consequences worth knowing:
            # the dumped HLO is a readable single layer, and the cold compile is short.
            #
            # `prevent_cse=False` under scan is deliberate. With it True, XLA is barred
            # from sharing work between the forward and the recomputation, which is
            # exactly the sharing remat depends on; the flag exists for the unrolled case
            # where CSE could otherwise delete the recomputation entirely.
            #
            # Layers stack on params axis 0 here. MaxText uses axis 1 (param_scan_axis),
            # so its dumped shapes read f32[4096,32,11008] where these read
            # f32[32,4096,11008]. Same parameters, different memory layout.
            stack = nn.scan(
                layer,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=nn.broadcast,
                length=cfg.n_layers,
            )
            hidden_states, _ = stack(cfg, name="layers")(hidden_states, position_ids)
        else:
            for i in range(cfg.n_layers):
                hidden_states, _ = layer(cfg, name=f"layers_{i}")(hidden_states, position_ids)

        hidden_states = LlamaRMSNorm(
            cfg.emb_dim,
            cfg.norm_eps,
            cfg.dtype,
            cfg.weight_dtype,
            name="norm",
        )(hidden_states)
        logits = nn.Dense(
            cfg.vocab_size,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
            name="lm_head",
        )(hidden_states)
        # fp32 for the softmax. The loss is a sum over 32000 logits and bf16 loses the
        # tail of that distribution; this is the second place precision has to go up.
        return logits.astype(jnp.float32)


# --------------------------------------------------------------------------- accounting


def analytic_tflops(cfg: ModelConfig, batch: int, seq: int) -> dict[str, float]:
    """The textbook FLOP count for one training step, per device.

    Reimplemented to match MaxText's `calculate_tflops_training_per_device` term for term,
    because it is the numerator of every MFU figure in this case study and the archived
    MaxText runs quote the same one. Checked against that run: 702.28 TFLOPs at batch 4,
    sequence 4096, which this reproduces.

    Three things about it are conventions rather than facts, and all three are why an MFU
    number is meaningless without its numerator stated:

      * The factor 3 is one forward pass plus two backward passes. It charges nothing for
        rematerialisation, so a run using an aggressive remat policy really executes more
        arithmetic than this counts.
      * The embedding lookup is a gather and is charged nothing; only the output
        projection is counted.
      * Attention is halved for causality, following Megatron-LM and NeMo.
    """
    b, t, e = batch, seq, cfg.emb_dim
    ffn = 6 * b * t * e * cfg.mlp_dim  # two up-projections plus one down, 2 FLOP per MAC
    qkv = 2 * b * t * e * (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim
    proj = 2 * b * t * e * cfg.n_heads * cfg.head_dim
    unembedding = 2 * b * t * e * cfg.vocab_size
    weight_flops = (ffn + qkv + proj) * cfg.n_layers + unembedding

    noncausal = 4 * b * t * t * cfg.n_heads * cfg.head_dim
    attention_flops = noncausal / 2 * cfg.n_layers

    learnable_tflops = weight_flops * 3 / 1e12
    attention_tflops = attention_flops * 3 / 1e12
    total = learnable_tflops + attention_tflops
    return {
        "total_tflops_per_step": total,
        "learnable_weight_tflops": learnable_tflops,
        "attention_tflops": attention_tflops,
        "learnable_weight_pct": 100 * learnable_tflops / total,
        "attention_pct": 100 * attention_tflops / total,
    }
