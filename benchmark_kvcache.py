import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from flash_attention_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache


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
    paged_block_size: Optional[int] = None


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


def prepare_dense_cache_for_reference(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    cache_batch_idx: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache_batch_idx is not None:
        k_cache = k_cache[cache_batch_idx.to(dtype=torch.long)]
        v_cache = v_cache[cache_batch_idx.to(dtype=torch.long)]
    if k_new is not None and v_new is not None:
        k_cache = k_cache.clone()
        v_cache = v_cache.clone()
        for batch_idx, start in enumerate(cache_seqlens.tolist()):
            k_cache[batch_idx, start : start + k_new.shape[1]] = k_new[batch_idx]
            v_cache[batch_idx, start : start + v_new.shape[1]] = v_new[batch_idx]
        effective_lengths = cache_seqlens + k_new.shape[1]
    else:
        effective_lengths = cache_seqlens
    max_len = int(effective_lengths.max().item()) if effective_lengths.numel() else 0
    return k_cache[:, :max_len], v_cache[:, :max_len], effective_lengths


def reference_attention_padded(
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
    k_used, v_used, effective_lengths = prepare_dense_cache_for_reference(
        k_cache,
        v_cache,
        cache_seqlens,
        k_new=k_new,
        v_new=v_new,
        cache_batch_idx=cache_batch_idx,
    )
    q_sdpa = q.permute(0, 2, 1, 3).contiguous()
    k_sdpa = k_used.permute(0, 2, 1, 3).contiguous()
    v_sdpa = v_used.permute(0, 2, 1, 3).contiguous()
    batch_size, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = k_sdpa.shape[2]
    k_pos = torch.arange(seqlen_k, device=q.device)[None, None, None, :]
    q_pos = torch.arange(seqlen_q, device=q.device)[None, None, :, None]
    lengths = effective_lengths.to(device=q.device)[:, None, None, None]
    attn_mask = k_pos < lengths
    if causal:
        attn_mask = attn_mask & (k_pos <= q_pos + lengths - seqlen_q)
    out = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=attn_mask,
        enable_gqa=q_sdpa.shape[1] != k_sdpa.shape[1],
        scale=softmax_scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def dense_flash_attention_workaround(
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
    """Pre-project workaround: pack all valid K/V tokens into a single varlen
    tensor and call flash_attn_varlen_func in one dispatch (no Python loop)."""
    k_used, v_used, effective_lengths = prepare_dense_cache_for_reference(
        k_cache,
        v_cache,
        cache_seqlens,
        k_new=k_new,
        v_new=v_new,
        cache_batch_idx=cache_batch_idx,
    )
    batch_size, seqlen_q, nheads_q, head_dim = q.shape
    # Pack Q: (batch, seqlen_q, nheads_q, head_dim) -> (batch*seqlen_q, nheads_q, head_dim)
    q_packed = q.reshape(batch_size * seqlen_q, nheads_q, head_dim)
    # Pack K/V: concatenate only valid tokens per batch element
    lengths = effective_lengths.tolist()
    k_packed = torch.cat([k_used[i, : lengths[i]] for i in range(batch_size)], dim=0)
    v_packed = torch.cat([v_used[i, : lengths[i]] for i in range(batch_size)], dim=0)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device
    )
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k[1:] = torch.cumsum(effective_lengths.to(torch.int32), dim=0)
    out_packed = flash_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=int(effective_lengths.max().item()),
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out_packed.reshape(batch_size, seqlen_q, nheads_q, head_dim)


def make_paged_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seqlen_cache, nheads_k, head_dim = k_cache.shape
    max_num_blocks_per_seq = (seqlen_cache + page_block_size - 1) // page_block_size
    num_blocks = batch_size * max_num_blocks_per_seq
    k_paged = torch.zeros(
        num_blocks,
        page_block_size,
        nheads_k,
        head_dim,
        device=k_cache.device,
        dtype=k_cache.dtype,
    )
    v_paged = torch.zeros_like(k_paged)
    block_table = torch.arange(num_blocks, device=k_cache.device, dtype=torch.int32).reshape(
        batch_size, max_num_blocks_per_seq
    )
    for batch_idx in range(batch_size):
        for block_idx in range(max_num_blocks_per_seq):
            start = block_idx * page_block_size
            end = min(start + page_block_size, seqlen_cache)
            if start >= end:
                continue
            physical_block = int(block_table[batch_idx, block_idx].item())
            k_paged[physical_block, : end - start].copy_(k_cache[batch_idx, start:end])
            v_paged[physical_block, : end - start].copy_(v_cache[batch_idx, start:end])
    return k_paged, v_paged, block_table


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


def global_gpu_warmup(device: torch.device, dtype: torch.dtype, *, iters: int) -> None:
    if iters <= 0:
        return
    torch.manual_seed(123)
    x = torch.randn((2048, 2048), device=device, dtype=dtype)
    y = torch.randn((2048, 2048), device=device, dtype=dtype)
    for _ in range(iters):
        torch.mm(x, y)
    torch.cuda.synchronize()


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


def run_case(
    case: BenchmarkCase,
    *,
    warmup: int,
    repeats: int,
    dtype: torch.dtype,
    device: torch.device,
    num_splits: int = 0,
    extra_baselines: bool = False,
) -> dict[str, Optional[float]]:
    q, k_cache, v_cache, cache_seqlens, cache_batch_idx, k_new, v_new = make_case_tensors(case, dtype, device)
    softmax_scale = q.shape[-1] ** (-0.5)
    block_table = None
    k_runtime = k_cache
    v_runtime = v_cache
    if case.paged_block_size is not None:
        k_runtime, v_runtime, block_table = make_paged_kvcache(
            k_cache, v_cache, case.paged_block_size
        )
        cache_batch_idx = None

    out = flash_attn_with_kvcache(
        q,
        k_runtime.clone(),
        v_runtime.clone(),
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens.clone(),
        cache_batch_idx=cache_batch_idx.clone() if cache_batch_idx is not None else None,
        block_table=block_table.clone() if block_table is not None else None,
        num_splits=num_splits,
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

    out_ref_padded = reference_attention_padded(
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
    torch.testing.assert_close(out_ref_padded, out_ref, atol=1e-2, rtol=1e-2)

    dense_flash_supported = True
    if dense_flash_supported:
        out_dense_flash = dense_flash_attention_workaround(
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
        torch.testing.assert_close(out_dense_flash, out_ref, atol=1e-2, rtol=1e-2)

    def run_flash() -> torch.Tensor:
        return flash_attn_with_kvcache(
            q,
            k_runtime,
            v_runtime,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            block_table=block_table,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=case.causal,
        )

    def run_reference_padded() -> torch.Tensor:
        return reference_attention_padded(
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

    def run_dense_flash() -> torch.Tensor:
        return dense_flash_attention_workaround(
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
    padded_median_ms = None
    dense_flash_median_ms = None
    if extra_baselines:
        padded_median_ms, _ = time_cuda(run_reference_padded, warmup=warmup, repeats=repeats)
        if dense_flash_supported:
            dense_flash_median_ms, _ = time_cuda(run_dense_flash, warmup=warmup, repeats=repeats)
    paged_tag = f" paged={case.paged_block_size}" if case.paged_block_size is not None else ""
    extra = ""
    if padded_median_ms is not None:
        extra += f" | padded_ref={padded_median_ms * 1000:.2f}us vs_flash={padded_median_ms / flash_median_ms:.2f}x"
    if dense_flash_median_ms is not None:
        extra += f" | dense_flash={dense_flash_median_ms * 1000:.2f}us vs_flash={dense_flash_median_ms / flash_median_ms:.2f}x"
    print(
        f"{case.name}{paged_tag} splits={num_splits}: "
        f"flash median={flash_median_ms * 1000:.2f}us mean={flash_mean_ms * 1000:.2f}us"
        f"{extra}"
    )
    return {
        "flash_median_ms": flash_median_ms,
        "flash_mean_ms": flash_mean_ms,
        "padded_ref_median_ms": padded_median_ms,
        "dense_flash_median_ms": dense_flash_median_ms,
    }


def run_split_sweep(
    case: BenchmarkCase,
    *,
    warmup: int,
    repeats: int,
    dtype: torch.dtype,
    device: torch.device,
    split_values: Sequence[int],
    extra_baselines: bool = False,
) -> Optional[dict]:
    best_split = None
    best_ms = float("inf")
    heuristic_ms = None
    padded_ref_ms = None
    dense_flash_ms = None
    for split in split_values:
        try:
            result = run_case(
                case,
                warmup=warmup,
                repeats=repeats,
                dtype=dtype,
                device=device,
                num_splits=split,
                extra_baselines=extra_baselines and split == 0,
            )
        except RuntimeError as err:
            print(f"{case.name}: skipping splits={split} due to runtime error: {err}")
            continue
        flash_median_ms = result["flash_median_ms"]
        if split == 0:
            heuristic_ms = flash_median_ms
            padded_ref_ms = result["padded_ref_median_ms"]
            dense_flash_ms = result["dense_flash_median_ms"]
        if flash_median_ms < best_ms:
            best_ms = flash_median_ms
            best_split = split
    if best_split is None:
        return None
    heuristic_delta_pct = 0.0
    if heuristic_ms is not None and heuristic_ms > 0.0:
        heuristic_delta_pct = ((heuristic_ms - best_ms) / heuristic_ms) * 100.0
    heuristic_tag = "n/a" if heuristic_ms is None else f"{heuristic_ms * 1000:.2f}us"
    best_vs_padded = None if padded_ref_ms is None else padded_ref_ms / best_ms
    heuristic_vs_padded = None if padded_ref_ms is None else padded_ref_ms / heuristic_ms
    best_vs_dense = None if dense_flash_ms is None else dense_flash_ms / best_ms
    heuristic_vs_dense = None if dense_flash_ms is None else dense_flash_ms / heuristic_ms
    print(
        f"{case.name}: best_split={best_split} best_flash_median={best_ms * 1000:.2f}us "
        f"heuristic_flash={heuristic_tag} best_vs_heuristic_delta={heuristic_delta_pct:.2f}%"
        f"{'' if best_vs_padded is None else f' best_vs_padded={best_vs_padded:.2f}x'}"
        f"{'' if best_vs_dense is None else f' best_vs_dense={best_vs_dense:.2f}x'}"
    )
    return {
        "case": case.name,
        "paged": case.paged_block_size is not None,
        "decode": case.seqlen_q == 1,
        "best_split": best_split,
        "best_vs_heuristic_delta": heuristic_delta_pct,
        "best_vs_padded": best_vs_padded,
        "heuristic_vs_padded": heuristic_vs_padded,
        "best_vs_dense": best_vs_dense,
        "heuristic_vs_dense": heuristic_vs_dense,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the report-quality KV-cache benchmark by default: split sweep, "
            "decode matrix, paged block size 256, and both timed baselines."
        )
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--split-sweep", dest="split_sweep", action="store_true", default=True)
    parser.add_argument("--no-split-sweep", dest="split_sweep", action="store_false")
    parser.add_argument("--split-max", type=int, default=8)
    parser.add_argument("--decode-matrix", dest="decode_matrix", action="store_true", default=True)
    parser.add_argument("--no-decode-matrix", dest="decode_matrix", action="store_false")
    parser.add_argument("--paged-block-sizes", type=str, default="256")
    parser.add_argument("--extra-baselines", dest="extra_baselines", action="store_true", default=True)
    parser.add_argument("--no-extra-baselines", dest="extra_baselines", action="store_false")
    parser.add_argument("--global-warmup-iters", type=int, default=80)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter smoke benchmark without split sweeps, decode matrix, paged cases, or baselines.",
    )
    args = parser.parse_args()

    if args.quick:
        args.warmup = min(args.warmup, 5)
        args.repeats = min(args.repeats, 10)
        args.split_sweep = False
        args.decode_matrix = False
        args.paged_block_sizes = ""
        args.extra_baselines = False
        args.global_warmup_iters = min(args.global_warmup_iters, 5)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmark_kvcache.py")

    device = torch.device("cuda")
    dtype = torch.float16
    global_gpu_warmup(device, dtype, iters=args.global_warmup_iters)
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
    if args.decode_matrix:
        cases.extend(
            [
                BenchmarkCase("read_decode_h64_ctx8k", 4, 4, 1, 8192, 0, 6, 1, 64, True, False, False),
                BenchmarkCase("read_decode_h64_ctx16k", 4, 4, 1, 16384, 0, 6, 1, 64, True, False, False),
                BenchmarkCase("read_decode_h128_ctx8k", 4, 4, 1, 8192, 0, 6, 1, 128, True, False, False),
                BenchmarkCase("read_decode_h128_ctx16k", 4, 4, 1, 16384, 0, 6, 1, 128, True, False, False),
            ]
        )

    paged_sizes = []
    if args.paged_block_sizes.strip():
        paged_sizes = [int(x) for x in args.paged_block_sizes.split(",") if x.strip()]
        paged_cases = []
        for block_size in paged_sizes:
            for base in cases:
                if base.has_batch_idx:
                    continue
                paged_cases.append(
                    BenchmarkCase(
                        name=f"{base.name}_paged",
                        batch_size=base.batch_size,
                        batch_size_cache=base.batch_size,
                        seqlen_q=base.seqlen_q,
                        seqlen_cache=base.seqlen_cache,
                        append_len=base.append_len,
                        nheads_q=base.nheads_q,
                        nheads_k=base.nheads_k,
                        head_dim=base.head_dim,
                        causal=base.causal,
                        append_new_kv=base.append_new_kv,
                        has_batch_idx=False,
                        paged_block_size=block_size,
                    )
                )
        cases.extend(paged_cases)

    print(
        f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
        f"warmup={args.warmup} repeats={args.repeats} global_warmup_iters={args.global_warmup_iters}"
    )
    if args.split_sweep:
        sweep_summaries = []
        split_values = list(range(0, max(args.split_max, 1) + 1))
        for case in cases:
            summary = run_split_sweep(
                case,
                warmup=args.warmup,
                repeats=args.repeats,
                dtype=dtype,
                device=device,
                split_values=split_values,
                extra_baselines=args.extra_baselines,
            )
            if summary is not None:
                sweep_summaries.append(summary)
        if sweep_summaries:
            groups = {
                "contiguous_decode": [r for r in sweep_summaries if (not r["paged"] and r["decode"])],
                "contiguous_chunk": [r for r in sweep_summaries if (not r["paged"] and not r["decode"])],
                "paged_decode": [r for r in sweep_summaries if (r["paged"] and r["decode"])],
                "paged_chunk": [r for r in sweep_summaries if (r["paged"] and not r["decode"])],
            }
            print("=== grouped_summary ===")
            for group_name, rows in groups.items():
                if not rows:
                    continue
                avg_best_delta = sum(r["best_vs_heuristic_delta"] for r in rows) / len(rows)
                split_hist = {}
                for r in rows:
                    split_hist[r["best_split"]] = split_hist.get(r["best_split"], 0) + 1
                extras = ""
                padded_rows = [r for r in rows if r["heuristic_vs_padded"] is not None]
                if padded_rows:
                    avg_best_vs_padded = sum(r["best_vs_padded"] for r in padded_rows) / len(padded_rows)
                    avg_heuristic_vs_padded = sum(r["heuristic_vs_padded"] for r in padded_rows) / len(padded_rows)
                    extras += (
                        f" avg_best_vs_padded={avg_best_vs_padded:.2f}x"
                        f" avg_heuristic_vs_padded={avg_heuristic_vs_padded:.2f}x"
                    )
                dense_rows = [r for r in rows if r["heuristic_vs_dense"] is not None]
                if dense_rows:
                    avg_best_vs_dense = sum(r["best_vs_dense"] for r in dense_rows) / len(dense_rows)
                    avg_heuristic_vs_dense = sum(r["heuristic_vs_dense"] for r in dense_rows) / len(dense_rows)
                    extras += (
                        f" avg_best_vs_dense={avg_best_vs_dense:.2f}x"
                        f" avg_heuristic_vs_dense={avg_heuristic_vs_dense:.2f}x"
                    )
                print(
                    f"{group_name}: n={len(rows)} avg_best_vs_heuristic_delta={avg_best_delta:.2f}% "
                    f"best_split_hist={split_hist}{extras}"
                )
    else:
        for case in cases:
            run_case(
                case,
                warmup=args.warmup,
                repeats=args.repeats,
                dtype=dtype,
                device=device,
                num_splits=0,
                extra_baselines=args.extra_baselines,
            )


if __name__ == "__main__":
    main()