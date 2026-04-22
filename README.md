# FlashAttention Turing

This repository provides an implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention) for the Turing architecture. 

## Features

Supports:

 - fwd and bwd
 - head dim 64, 128
 - causal mask
 - gqa
 - contiguous inference-only kv cache
 - varlen

Does not support:

 - dropout
 - local mask
 - kv cache backward
 - paged kv cache
 - rotary / alibi / leftpad in kv cache mode

## Performance

We currently have benchmarks for T4.

### Forward pass 

Up to 2.19x and 1.95x faster than PyTorch's [Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for non-causal and causal workloads.

On Turing GPUs, PyTorch's Attention calls Memory-Efficient Attention from [xformers](https://github.com/facebookresearch/xformers) in the backend.

For long sequences, the forward kernel reaches up to 66% compute throughput.

<img src="utils/forward_128_combined.png" alt="Forward pass benchmark for head dimension 128" width="1000">
<img src="utils/forward_64_combined.png" alt="Forward pass benchmark for head dimension 64" width="1000">

### Backward pass 

The backward pass is split into two kernels: one for `dQ` and one for `dK` and `dV`.

Up to 1.35x and 1.51x faster than PyTorch's Attention for non-causal and causal workloads.

For long sequences, the backward kernels reach up to 49% compute throughput for `dK` and `dV`, and 45% for `dQ`.

<img src="utils/backward_128_combined.png" alt="Backward pass benchmark for head dimension 128" width="1000">
<img src="utils/backward_64_combined.png" alt="Backward pass benchmark for head dimension 64" width="1000">


## How to use FlashAttention
The main functions implement scaled dot product attention: `softmax(Q @ K^T * softmax_scale) @ V`.

```
from flash_attention_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
```


The arguments for these functions differ from the standard FlashAttention Python API because this implementation does not support every feature yet. See `turing/flash_attention_interface.py` for the full function signatures and parameter descriptions.

`flash_attn_with_kvcache` currently supports contiguous cache tensors only and is intended for inference. If `k` and `v` are provided, they are appended into `k_cache` and `v_cache` in place starting at `cache_seqlens`, then attention runs against the updated cache. The API also supports `cache_batch_idx` to remap query rows onto cache rows. This path does not provide backward support and still keeps a smaller feature set than the full mainline FlashAttention KV-cache API.


## Requirements
We tested this implementation with:

- CUDA 12.4
- PyTorch 2.8.0 and 2.5.1

## Build notes

Install with:

```bash
git clone https://github.com/ssiu/flash-attention-turing
cd /path/to/flash-attention-turing
pip install torch setuptools ninja wheel
pip install -v .
```

To run the test suite, install the test dependencies:

```bash
pip install pytest numpy pandas
pytest -q
```

If you enable Excel debug dumps in `test_flash_attn.py`, also install:

```bash
pip install openpyxl
```
