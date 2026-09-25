"""Train Llama 2 7B on one GPU, in JAX. The whole loop, with nothing behind a framework.

    python3 model/train.py configs/jax.yml steps=30

Standalone scripts select the precision, attention implementation, remat policy, and
profiling mode. This file owns only the training step and its log output.

WHAT THIS DELIBERATELY DOES NOT HAVE

No checkpointing, no evaluation, no data loader, no sharding, no distributed anything. The
data is synthetic and reused, which removes the input pipeline from every measurement: a
step time here is an upper bound on the same step with a real loader attached.

THE MEASUREMENT CONTRACT

The step lines match MaxText so the two implementations can be compared directly.

    completed step: N, seconds: S, TFLOP/s/device: T, Tokens/s/device: K, loss: L
    Per train step:\\n Total TFLOPs: X\\n split as A% learnable weight flops and B% ...
    Total memory size: X GB, Output size: ..., Temp size: ..., Argument size: ..., ...
    Memstats: <label>:\\n Using (GB) U / L (P%) on <device>
    number parameters: N billion

Logging goes through absl so every step has a timestamp.
"""

from __future__ import annotations

import ctypes
import functools
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import yaml
from absl import app, logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama7b import Llama, ModelConfig, analytic_tflops  # noqa: E402

GB = 1024**3

# The measurement contract, as literal format strings, in one place.
LOG_PARAMS = "number parameters: %.3f billion"
LOG_MEMSTATS_LABEL = "\nMemstats: %s:"
LOG_MEMSTATS_USED = "\tUsing (GB) %.2f / %.2f (%f%%) on %s"
LOG_MEMSTATS_PEAK = "\tPeak (GB) %.2f on %s"
LOG_COMPILED_MEM = (
    "Total memory size: %.1f GB, Output size: %.1f GB, Temp size: %.1f GB, "
    "Argument size: %.1f GB, Host temp size: %.1f GB."
)
LOG_LOOP_START = "====== Starting training loop ======"
LOG_STEP = "completed step: %d, seconds: %.3f, TFLOP/s/device: %.3f, Tokens/s/device: %.3f, loss: %.3f"
# Deliberately not through absl: this one is multi-line and the prefix would land only on
# the first line, which is exactly the shape MaxText's own print produced and what run.py's
# regex was written against.
LOG_ANALYTIC_FLOPS = (
    "Per train step:\n Total TFLOPs: %.2f \n split as %.2f%% learnable weight flops and %.2f%% attention flops"
)


# --------------------------------------------------------------------------- config


def load_config(path: Path, overrides: dict[str, str]) -> dict[str, Any]:
    """Read a yaml config, follow one level of `base_config`, then apply key=value args.

    Ten lines instead of a config framework. The merge order is
    base -> config -> overrides, which is the same order MaxText uses, minus the third
    precedence rule that made an architectural key in a base file silently unreachable.
    """
    cfg: dict[str, Any] = {}
    path = path.resolve()
    loaded = yaml.safe_load(path.read_text()) or {}
    base = loaded.pop("base_config", None)
    if base:
        base_path = (path.parent / base).resolve()
        if not base_path.is_file():
            raise SystemExit(f"{path}: base_config {base!r} not found at {base_path}")
        cfg.update(yaml.safe_load(base_path.read_text()) or {})
        cfg.pop("base_config", None)
    cfg.update(loaded)

    for key, raw in overrides.items():
        if key not in cfg:
            raise SystemExit(f"unknown config key in override: {key}={raw}. Add it to the yaml first.")
        cfg[key] = _coerce(raw, cfg[key])
    return cfg


def _coerce(raw: str, current: Any) -> Any:
    """Take the type from the value already in the config, so overrides cannot retype a key."""
    if isinstance(current, bool):
        return str(raw).lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw) if raw != "" else 0
    if isinstance(current, float):
        return float(raw)
    return raw


DTYPES = {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "float16": jnp.float16}


def model_config(cfg: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        emb_dim=cfg["emb_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        mlp_dim=cfg["mlp_dim"],
        vocab_size=cfg["vocab_size"],
        norm_eps=float(cfg["norm_eps"]),
        rope_max_timescale=cfg["rope_max_timescale"],
        dtype=DTYPES[cfg["dtype"]],
        weight_dtype=DTYPES[cfg["weight_dtype"]],
        attention=cfg["attention"],
        remat_policy=cfg["remat_policy"],
        scan_layers=cfg["scan_layers"],
    )


# --------------------------------------------------------------------------- training


def build_optimizer(cfg: dict[str, Any]) -> optax.GradientTransformation:
    """AdamW with fp32 moments, global-norm clipping, warmup then cosine decay.

    `mu_dtype` is pinned to float32 rather than inherited. Adam's first moment in bf16 is
    a real and popular memory saving -- it is 25 GiB at this model size -- and it is also
    a change of numerics, so it is not something to acquire by default. plan.md freezes
    the precision recipe and this is part of it.
    """
    total = cfg["learning_rate_schedule_steps"]
    warmup = max(1, int(total * cfg["warmup_steps_fraction"]))
    schedule = optax.join_schedules(
        [
            optax.linear_schedule(0.0, cfg["learning_rate"], warmup),
            optax.cosine_decay_schedule(
                cfg["learning_rate"], max(1, total - warmup), alpha=cfg["learning_rate_final_fraction"]
            ),
        ],
        [warmup],
    )
    return optax.chain(
        optax.clip_by_global_norm(cfg["gradient_clipping_threshold"]),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg["adam_b1"],
            b2=cfg["adam_b2"],
            eps=float(cfg["adam_eps"]),
            weight_decay=cfg["adam_weight_decay"],
            mu_dtype=jnp.float32,
        ),
    )


def loss_fn(params, model, batch):
    """Next-token cross entropy, averaged over tokens.

    The logits arrive in fp32 (see Llama.__call__) and the log-softmax stays there. That
    is not free: [4, 4096, 32000] in fp32 is 2 GiB, which makes the logits the single
    largest activation in the step. Worth knowing before blaming attention for the memory
    profile.

    Integer labels rather than one-hot, and the difference is not stylistic. A one-hot
    target tensor is the same 2 GiB again, and the multiply against it is 500M FLOPs of
    reading zeroes. `softmax_cross_entropy_with_integer_labels` gathers the target logit
    instead, which is the same arithmetic without the tensor.
    """
    logits = model.apply(params, batch["inputs"], batch["positions"])
    return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, batch["targets"]))


def make_train_step(model, optimizer):
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def synthetic_batch(key, batch: int, seq: int, vocab: int) -> dict[str, jax.Array]:
    """One batch of random tokens, reused for every step.

    Reusing it is the point: it removes the input pipeline, host-to-device transfer and
    any dataset variance from the measurement, leaving the compiled step. It also means
    the loss falls to near zero within thirty steps, which is not a bug and not a result
    -- the model is memorising one batch. Convergence is explicitly out of scope.
    """
    tokens = jax.random.randint(key, (batch, seq + 1), 0, vocab, dtype=jnp.int32)
    return {
        "inputs": tokens[:, :-1],
        "targets": tokens[:, 1:],
        "positions": jnp.arange(seq, dtype=jnp.int32)[None, :].repeat(batch, axis=0),
    }


# --------------------------------------------------------------------------- reporting


def log_memstats(label: str) -> None:
    """Allocator high-water, in the format run.py parses."""
    logging.info(LOG_MEMSTATS_LABEL, label)
    for d in jax.local_devices():
        try:
            s = d.memory_stats()
            used, limit = s["bytes_in_use"] / GB, s["bytes_limit"] / GB
            logging.info(LOG_MEMSTATS_USED, used, limit, used / limit * 100, d)
            if "peak_bytes_in_use" in s:
                logging.info(LOG_MEMSTATS_PEAK, s["peak_bytes_in_use"] / GB, d)
        except (RuntimeError, KeyError, TypeError) as e:
            logging.info("\tMemstats unavailable, error: %s", e)


def log_compiled_memory(compiled) -> None:
    """XLA's static analysis of the executable: the buffer plan, before a byte is allocated.

    This is the figure the experiment's memory estimate predicts, so it is
    reported in the same units and the same wording MaxText used. Note `alias` is
    subtracted: donated buffers are counted once, not twice.
    """
    s = compiled.memory_analysis()
    if s is None:
        return
    out, tmp, arg = s.output_size_in_bytes / GB, s.temp_size_in_bytes / GB, s.argument_size_in_bytes / GB
    alias, host = s.alias_size_in_bytes / GB, s.host_temp_size_in_bytes / GB
    logging.info(LOG_COMPILED_MEM, out + tmp + arg - alias, out, tmp, arg, host)


def log_analytic_flops(flops: dict[str, float]) -> None:
    print(
        LOG_ANALYTIC_FLOPS
        % (flops["total_tflops_per_step"], flops["learnable_weight_pct"], flops["attention_pct"]),
        flush=True,
    )


def rocprof_selected_step() -> tuple[int, ctypes.CDLL] | None:
    """Pause rocprof until one requested training step.

    Counter collection serializes GPU dispatches. Selecting one warmed-up step
    keeps parameter initialization and the other loop iterations out of the
    counter totals while preserving the unprofiled experiment behavior.
    """
    raw = os.environ.get("ROCPROF_SELECTED_STEP")
    if raw is None:
        return None

    step = int(raw)
    roctx = ctypes.CDLL("librocprofiler-sdk-roctx.so.1")
    for name in ("roctxProfilerPause", "roctxProfilerResume"):
        function = getattr(roctx, name)
        function.argtypes = [ctypes.c_uint64]
        function.restype = ctypes.c_int
    if roctx.roctxProfilerPause(0) != 0:
        raise RuntimeError("rocprofiler rejected the initial pause request")
    return step, roctx


# --------------------------------------------------------------------------- entry point


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    config_path = Path(argv[1])
    overrides = dict(a.split("=", 1) for a in argv[2:] if "=" in a)
    cfg = load_config(config_path, overrides)
    selected_profile = rocprof_selected_step()
    if selected_profile is not None and selected_profile[0] >= cfg["steps"]:
        raise SystemExit(
            f"ROCPROF_SELECTED_STEP={selected_profile[0]} requires steps>{selected_profile[0]}"
        )

    # Cold compile unless a cache is explicitly requested. An empty value is the default
    # and is what run.py always passes: a warm cache silently reuses an executable built
    # under different compiler flags, which is how two runs of the predecessor project
    # ended up incomparable.
    if cfg.get("jax_cache_dir"):
        jax.config.update("jax_compilation_cache_dir", cfg["jax_cache_dir"])

    logging.info("System Information: Jax Version: %s", jax.__version__)
    logging.info("System Information: Jax Backend: %s", jax.devices()[0].client.platform_version)
    logging.info("System Information: Number of devices: %d, devices: %s", jax.device_count(), jax.devices())

    mcfg = model_config(cfg)
    batch, seq = int(cfg["per_device_batch_size"]), cfg["max_target_length"]
    model = Llama(mcfg)
    optimizer = build_optimizer(cfg)

    key = jax.random.key(cfg["seed"])
    init_key, data_key = jax.random.split(key)
    example = synthetic_batch(data_key, batch, seq, mcfg.vocab_size)

    params = jax.jit(model.init)(init_key, example["inputs"], example["positions"])
    n_params = sum(x.size for x in jax.tree.leaves(params))
    logging.info(LOG_PARAMS, n_params / 1e9)
    log_memstats("After params initialized")

    opt_state = jax.jit(optimizer.init)(params)
    log_memstats("After optimizer initialized")

    flops = analytic_tflops(mcfg, batch, seq)
    log_analytic_flops(flops)

    # Compiled ahead of the loop rather than lazily on the first step, so that compile time
    # is attributable and step 0 measures a step. Donating params and opt_state is what
    # lets XLA update them in place; without it the peak holds two copies of a 100 GiB
    # optimizer state and the model does not fit.
    step_fn = make_train_step(model, optimizer)
    compiled = jax.jit(step_fn, donate_argnums=(0, 1)).lower(params, opt_state, example).compile()
    log_compiled_memory(compiled)

    profiler_on = cfg.get("profiler") == "xplane"
    trace_dir = Path(cfg["base_output_directory"]) / "trace"
    profile_from = cfg["skip_first_n_steps_for_profiler"]
    profile_to = profile_from + cfg["profiler_steps"]

    tokens_per_step = batch * seq
    logging.info(LOG_LOOP_START)

    for step in range(cfg["steps"]):
        if selected_profile is not None and step == selected_profile[0]:
            if selected_profile[1].roctxProfilerResume(0) != 0:
                raise RuntimeError("rocprofiler rejected the resume request")
        if profiler_on and step == profile_from:
            trace_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(trace_dir))
        t0 = time.perf_counter()
        params, opt_state, loss = compiled(params, opt_state, example)
        # JAX dispatches asynchronously, so without this the loop measures how fast Python
        # can enqueue work rather than how fast the GPU can do it.
        jax.block_until_ready((params, opt_state, loss))
        dt = time.perf_counter() - t0
        if selected_profile is not None and step == selected_profile[0]:
            if selected_profile[1].roctxProfilerPause(0) != 0:
                raise RuntimeError("rocprofiler rejected the final pause request")
        if profiler_on and step == profile_to - 1:
            jax.profiler.stop_trace()

        logging.info(
            LOG_STEP, step, dt, flops["total_tflops_per_step"] / dt, tokens_per_step / dt, float(loss)
        )

    log_memstats("After training loop")


if __name__ == "__main__":
    app.run(main)
