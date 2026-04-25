import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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
) -> tuple[float, float, float, float]:
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
    paged_tag = f" paged={case.paged_block_size}" if case.paged_block_size is not None else ""
    print(
        f"{case.name}{paged_tag} splits={num_splits}: "
        f"flash median={flash_median_ms * 1000:.2f}us mean={flash_mean_ms * 1000:.2f}us | "
        f"ref median={ref_median_ms * 1000:.2f}us mean={ref_mean_ms * 1000:.2f}us | speedup={speedup:.2f}x"
    )
    return flash_median_ms, flash_mean_ms, ref_median_ms, ref_mean_ms


def run_split_sweep(
    case: BenchmarkCase,
    *,
    warmup: int,
    repeats: int,
    dtype: torch.dtype,
    device: torch.device,
    split_values: Sequence[int],
) -> Optional[dict]:
    best_split = None
    best_ms = float("inf")
    baseline_ref_ms = None
    heuristic_ms = None
    for split in split_values:
        try:
            flash_median_ms, _, ref_median_ms, _ = run_case(
                case,
                warmup=warmup,
                repeats=repeats,
                dtype=dtype,
                device=device,
                num_splits=split,
            )
        except RuntimeError as err:
            print(f"{case.name}: skipping splits={split} due to runtime error: {err}")
            continue
        if split == 0:
            heuristic_ms = flash_median_ms
        if baseline_ref_ms is None:
            baseline_ref_ms = ref_median_ms
        if flash_median_ms < best_ms:
            best_ms = flash_median_ms
            best_split = split
    if baseline_ref_ms is None or best_split is None:
        return None
    best_speedup = baseline_ref_ms / best_ms
    heuristic_delta_pct = 0.0
    if heuristic_ms is not None and heuristic_ms > 0.0:
        heuristic_delta_pct = ((heuristic_ms - best_ms) / heuristic_ms) * 100.0
    heuristic_tag = "n/a" if heuristic_ms is None else f"{heuristic_ms * 1000:.2f}us"
    print(
        f"{case.name}: best_split={best_split} best_flash_median={best_ms * 1000:.2f}us "
        f"best_speedup={best_speedup:.2f}x heuristic_flash={heuristic_tag} "
        f"best_vs_heuristic_delta={heuristic_delta_pct:.2f}%"
    )
    return {
        "case": case.name,
        "paged": case.paged_block_size is not None,
        "decode": case.seqlen_q == 1,
        "best_split": best_split,
        "best_speedup": best_speedup,
        "best_vs_heuristic_delta": heuristic_delta_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--split-sweep", action="store_true")
    parser.add_argument("--split-max", type=int, default=8)
    parser.add_argument("--decode-matrix", action="store_true")
    parser.add_argument("--paged-block-sizes", type=str, default="")
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

    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} warmup={args.warmup} repeats={args.repeats}")
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
                avg_best_speedup = sum(r["best_speedup"] for r in rows) / len(rows)
                avg_best_delta = sum(r["best_vs_heuristic_delta"] for r in rows) / len(rows)
                split_hist = {}
                for r in rows:
                    split_hist[r["best_split"]] = split_hist.get(r["best_split"], 0) + 1
                print(
                    f"{group_name}: n={len(rows)} avg_best_speedup={avg_best_speedup:.2f}x "
                    f"avg_best_vs_heuristic_delta={avg_best_delta:.2f}% best_split_hist={split_hist}"
                )
    else:
        for case in cases:
            run_case(case, warmup=args.warmup, repeats=args.repeats, dtype=dtype, device=device, num_splits=0)


if __name__ == "__main__":
    main()