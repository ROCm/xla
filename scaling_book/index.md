---
layout: distill
title: "How to Scale Your Model on AMD"
subtitle: "An AMD companion to How to Scale Your Model, performed on MI355X"
description: "How MI355X hardware, ROCm kernels, JAX compilation, and MaxText configuration affect training throughput."
date: 2026-09-10

section_number: 0

previous_section_url: ""
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/1-mi355x"
next_section_name: "Chapter 1: MI355X architecture and system topology"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Motivation"
  - name: "Reader and Prerequisites"
  - name: "Chapters"
    subsections:
      - name: "1. MI355X architecture and system topology"
        url: "/pages/1-mi355x"
      - name: "2. The JAX/ROCm Stack"
        url: "/pages/2-jax-rocm-stack"
      - name: "3. Profiling a Training Step"
        url: "/pages/3-profiling"
      - name: "4. Training in Mixed Precision"
        url: "/pages/4-mixed-precision"
      - name: "5. Sharding and Parallelism"
        url: "/pages/5-sharding"
      - name: "6. Memory and Kernel Optimizations"
        url: "/pages/6-mem-and-kernel-optimizations"
  - name: "Attribution"
---

## Motivation

The renowned [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) teaches a comprehensive understanding on TPU architecure, rooflines, Transformer analysis, sharding behaviour, training parallelism, and much more.

This ROCm book serves as complementary material for readers to understand the ROCm stack for JAX, including ROCm experimental features, and in general how to extract the best training performance out of AMD GPU's with JAX.

But perhaps more importantly, readers should be aware of the performance features available within the JAX/XLA/MaxText ecosystem, understand their underlying mechanisms, and know where to find further information.

Given the pace of active development, the JAX/ROCm software stack evolves rapidly, with new optimizations and capabilities often arriving faster than documentation can be updated. Consequently, the goal is not to present current performance recommendations, but to provide readers with the context necessary to discover new developments, evaluate their applicability, and incorporate optimizations suited to their own training environments.

## Reader and Prerequisites

The intended reader knows Python, basic JAX, and Transformer block architecture. The
book introduces AMD hardware, the ROCm execution path, and the sharding and
profiling details needed by the experiments. Readers who want first-principles
explanations and derivations should read the JAX Scaling Book chapters linked throughout.

## Chapters

1. [**MI355X architecture and system topology**]({{ '/pages/1-mi355x' | relative_url }})
   explains CDNA 4 matrix execution, memory, native precision formats, and AMD's Infinity Fabric topology.
2. [**The JAX/ROCm Stack**]({{ '/pages/2-jax-rocm-stack' | relative_url }})
   follows a JAX program through tracing, StableHLO, XLA, ROCm libraries, FFI,
   and device execution.
3. [**Profiling a Training Step**]({{ '/pages/3-profiling' | relative_url }})
   connects roofline estimates to XProf, `rocprofv3`, hardware counters, and
   `rocprof-compute`.
4. [**Training in Mixed Precision**]({{ '/pages/4-mixed-precision' | relative_url }})
   explains training in BF16, FP16, FP8, MXFP8, MXFP6, and MXFP4, accompanied with case studies on Llama 70B.
5. [**Sharding and Parallelism**]({{ '/pages/5-sharding' | relative_url }})
   explains different parallelism strategies through sharding, accompanied with case studies on Mixtral 8x22B
6. [**Memory and Kernel Optimizations**]({{ '/pages/6-mem-and-kernel-optimizations' | relative_url }})
   covers rematerialization, ROCm attention backends, and grouped GEMM lowering
   for MoE models.

## Attribution

The book reuses concepts and, where noted, adapted material from the MIT-licensed
JAX Scaling Book. Citations accompany reused derivations and figures. AMD, JAX,
OpenXLA, and OCP specifications are cited where their facts are used.

<h3 markdown=1 class="next-section">Next: [Chapter 1, MI355X architecture and system topology]({{ '/pages/1-mi355x' | relative_url }}).</h3>
