# JAX Scaling Book for AMD GPU's

This book is available publicly at [rocm.github.io/xla/scaling-book](https://rocm.github.io/xla/scaling-book/) and live edits will first be visible at this URL.

The book is based off [Google's JAX Scaling Book](https://jax-ml.github.io/scaling-book/) and is intended for clients/consumers of AMD GPUs, or otherwise interested readers, to understand the performance features available for training in JAX. It covers content mostly at the implementation layer, such as ROCm libraries, MaxText, XLA internals, and JAX APIs.

The repo for reproduction of the documented case studies can be found at:

- [Llama 7B Rematerialisation and Attention Backend Experiments (and JAX profiler + rocprofv3 examples)](https://github.com/clarkechong/llama7b-jax-fundamentals)
- [Llama 70B Mixed Precision Training Experiments](https://github.com/clarkechong/llama70b-mixed-precision-training)
- [Mixtral 8x22B Sharding and Grouped GEMM Experiments](https://github.com/clarkechong/mixtral8-22b-distributed-strategies)

If reproducing from these repos, it would be recommended to adjust the training steps per experiment to fit your GPU time budget. These experiments were run on an 8x MI355X node but should be extensible to any CDNA architectures supported by XLA.

### Maintenance

- Edits will be made to the GitHub page over time, and the intention is to open PRs from time to time in order to sync this internal copy with the live book.
- This is still essentially a first version of the book. Feel free to modify this internal copy to be more accurate or precise in any appropriate ways.
