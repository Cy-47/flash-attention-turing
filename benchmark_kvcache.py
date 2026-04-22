import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from flash_attention_interface import flash_attn_with_kvcache


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch_size: int
    batch_size_cache: int
    seqlen_q: int
    seqlen_cache: int
    append_len: int
    nheads_q: int
    nheads_k: int
    head_dim: int
    causal: bool
    append_new_kv: bool
    has_batch_idx: bool


def causal_lower_right(seqlen_q: int, seqlen_k: int, device: torch.device) -> torch.Tensor:
    diagonal_offset = seqlen_k - seqlen_q
    return torch.tril(
        torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device),
        diagonal=diagonal_offset,
    )


def reference_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    cache_batch_idx: Optional[torch.Tensor],
    causal: bool,
    softmax_scale: Optional[float],
) -> torch.Tensor:
    if cache_batch_idx is not None:
        k_cache = k_cache[cache_batch_idx.to(dtype=torch.long)]
        v_cache = v_cache[cache_batch_idx.to(dtype=torch.long)]
    if k_new is not None and v_new is not None:
        for batch_idx, start in enumerate(cache_seqlens.tolist()):
            k_cache[batch_idx, start : start + k_new.shape[1]] = k_new[batch_idx]
            v_cache[batch_idx, start : start + v_new.shape[1]] = v_new[batch_idx]
        effective_lengths = cache_seqlens + k_new.shape[1]
    else:
        effective_lengths = cache_seqlens

    outputs = []
    for batch_idx, seqlen_k in enumerate(effective_lengths.tolist()):
        q_i = q[batch_idx : batch_idx + 1].permute(0, 2, 1, 3).contiguous()
        k_i = k_cache[batch_idx : batch_idx + 1, :seqlen_k].permute(0, 2, 1, 3).contiguous()
        v_i = v_cache[batch_idx : batch_idx + 1, :seqlen_k].permute(0, 2, 1, 3).contiguous()
        attn_mask = None
        is_causal = False
        if causal:
            if q_i.shape[2] == k_i.shape[2]:
                is_causal = True
            else:
                attn_mask = causal_lower_right(q_i.shape[2], k_i.shape[2], q.device)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=q_i.shape[1] != k_i.shape[1],
            scale=softmax_scale,
        )
        outputs.append(out_i.permute(0, 2, 1, 3).contiguous())
    return torch.cat(outputs, dim=0)


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))
    return statistics.median(timings_ms), statistics.mean(timings_ms)


def make_case_tensors(case: BenchmarkCase, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    q = torch.randn(case.batch_size, case.seqlen_q, case.nheads_q, case.head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(case.batch_size_cache, case.seqlen_cache, case.nheads_k, case.head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(case.batch_size_cache, case.seqlen_cache, case.nheads_k, case.head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor(
        [case.seqlen_cache // 2 + 13 * idx for idx in range(case.batch_size)],
        device=device,
        dtype=torch.int32,
    )
    cache_batch_idx = (
        torch.randperm(case.batch_size_cache, dtype=torch.int32, device=device)[: case.batch_size]
        if case.has_batch_idx
        else None
    )
    if case.append_new_kv:
        k_new = torch.randn(case.batch_size, case.append_len, case.nheads_k, case.head_dim, device=device, dtype=dtype)
        v_new = torch.randn(case.batch_size, case.append_len, case.nheads_k, case.head_dim, device=device, dtype=dtype)
    else:
        k_new = None
        v_new = None
    return q, k_cache, v_cache, cache_seqlens, cache_batch_idx, k_new, v_new


def run_case(case: BenchmarkCase, *, warmup: int, repeats: int, dtype: torch.dtype, device: torch.device) -> None:
    q, k_cache, v_cache, cache_seqlens, cache_batch_idx, k_new, v_new = make_case_tensors(case, dtype, device)
    softmax_scale = q.shape[-1] ** (-0.5)

    out = flash_attn_with_kvcache(
        q,
        k_cache.clone(),
        v_cache.clone(),
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens.clone(),
        cache_batch_idx=cache_batch_idx.clone() if cache_batch_idx is not None else None,
        softmax_scale=softmax_scale,
        causal=case.causal,
    )
    out_ref = reference_attention(
        q,
        k_cache.clone(),
        v_cache.clone(),
        cache_seqlens.clone(),
        k_new=k_new,
        v_new=v_new,
        cache_batch_idx=cache_batch_idx.clone() if cache_batch_idx is not None else None,
        causal=case.causal,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)

    def run_flash() -> torch.Tensor:
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            softmax_scale=softmax_scale,
            causal=case.causal,
        )

    def run_reference() -> torch.Tensor:
        return reference_attention(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            k_new=k_new,
            v_new=v_new,
            cache_batch_idx=cache_batch_idx,
            causal=case.causal,
            softmax_scale=softmax_scale,
        )

    flash_median_ms, flash_mean_ms = time_cuda(run_flash, warmup=warmup, repeats=repeats)
    ref_median_ms, ref_mean_ms = time_cuda(run_reference, warmup=warmup, repeats=repeats)
    speedup = ref_median_ms / flash_median_ms
    print(
        f"{case.name}: flash median={flash_median_ms * 1000:.2f}us mean={flash_mean_ms * 1000:.2f}us | "
        f"ref median={ref_median_ms * 1000:.2f}us mean={ref_mean_ms * 1000:.2f}us | speedup={speedup:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmark_kvcache.py")

    device = torch.device("cuda")
    dtype = torch.float16
    cases = [
        BenchmarkCase("read_decode_h64_causal", 2, 2, 1, 4096, 0, 6, 1, 64, True, False, False),
        BenchmarkCase("append_decode_h64_causal", 2, 2, 1, 4096, 1, 6, 1, 64, True, True, False),
        BenchmarkCase("append_decode_h64_causal_batchidx", 2, 4, 1, 4096, 1, 6, 1, 64, True, True, True),
        BenchmarkCase("read_chunk_h64_noncausal", 2, 2, 16, 4096, 0, 4, 2, 64, False, False, False),
        BenchmarkCase("append_chunk_h64_noncausal", 2, 2, 16, 4096, 3, 4, 2, 64, False, True, False),
        BenchmarkCase("append_chunk_h64_noncausal_batchidx", 2, 4, 16, 4096, 3, 4, 2, 64, False, True, True),
        BenchmarkCase("read_decode_h128_causal", 2, 2, 1, 4096, 0, 6, 1, 128, True, False, False),
        BenchmarkCase("append_decode_h128_causal", 2, 2, 1, 4096, 1, 6, 1, 128, True, True, False),
        BenchmarkCase("append_decode_h128_causal_batchidx", 2, 4, 1, 4096, 1, 6, 1, 128, True, True, True),
        BenchmarkCase("read_chunk_h128_noncausal", 2, 2, 16, 4096, 0, 4, 2, 128, False, False, False),
        BenchmarkCase("append_chunk_h128_noncausal", 2, 2, 16, 4096, 3, 4, 2, 128, False, True, False),
        BenchmarkCase("append_chunk_h128_noncausal_batchidx", 2, 4, 16, 4096, 3, 4, 2, 128, False, True, True),
    ]

    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} warmup={args.warmup} repeats={args.repeats}")
    for case in cases:
        run_case(case, warmup=args.warmup, repeats=args.repeats, dtype=dtype, device=device)


if __name__ == "__main__":
    main()