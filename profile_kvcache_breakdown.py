import argparse
import statistics
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from benchmark_kvcache import (
    BenchmarkCase,
    global_gpu_warmup,
    make_case_tensors,
    make_paged_kvcache,
    prepare_dense_cache_for_reference,
)
from flash_attention_interface import flash_attn_varlen_func, flash_attn_with_kvcache


def measure(fn: Callable[[], object], *, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cuda_ms = []
    wall_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cuda_ms.append(start.elapsed_time(end))
        wall_ms.append((t1 - t0) * 1000.0)
    return statistics.median(cuda_ms), statistics.median(wall_ms)


def print_row(case_name: str, component: str, cuda_ms: float, wall_ms: float) -> None:
    print(
        f"BREAKDOWN case={case_name} component={component} "
        f"cuda_us={cuda_ms * 1000.0:.2f} wall_us={wall_ms * 1000.0:.2f}",
        flush=True,
    )


def print_scaling(
    sweep: str,
    case_name: str,
    variable: str,
    value: int,
    component: str,
    cuda_ms: float,
    wall_ms: float,
) -> None:
    print(
        f"SCALING sweep={sweep} case={case_name} {variable}={value} component={component} "
        f"cuda_us={cuda_ms * 1000.0:.2f} wall_us={wall_ms * 1000.0:.2f}",
        flush=True,
    )


def make_dense_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    cache_batch_idx: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_used, v_used, effective_lengths = prepare_dense_cache_for_reference(
        k_cache,
        v_cache,
        cache_seqlens,
        k_new=k_new,
        v_new=v_new,
        cache_batch_idx=cache_batch_idx,
    )
    return q, k_used, v_used, effective_lengths


def make_varlen_inputs(
    q: torch.Tensor,
    k_used: torch.Tensor,
    v_used: torch.Tensor,
    effective_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    batch_size, seqlen_q, nheads_q, head_dim = q.shape
    q_packed = q.reshape(batch_size * seqlen_q, nheads_q, head_dim)
    lengths = effective_lengths.tolist()
    k_packed = torch.cat([k_used[i, : lengths[i]] for i in range(batch_size)], dim=0)
    v_packed = torch.cat([v_used[i, : lengths[i]] for i in range(batch_size)], dim=0)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device
    )
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k[1:] = torch.cumsum(effective_lengths.to(torch.int32), dim=0)
    return (
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        int(effective_lengths.max().item()),
    )


def make_sdpa_inputs(
    q: torch.Tensor,
    k_used: torch.Tensor,
    v_used: torch.Tensor,
    effective_lengths: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return q_sdpa, k_sdpa, v_sdpa, attn_mask


def profile_case(case: BenchmarkCase, *, warmup: int, repeats: int, device: torch.device) -> None:
    dtype = torch.float16
    q, k_cache, v_cache, cache_seqlens, cache_batch_idx, k_new, v_new = make_case_tensors(
        case, dtype, device
    )
    softmax_scale = q.shape[-1] ** (-0.5)
    block_table = None
    k_runtime = k_cache
    v_runtime = v_cache
    if case.paged_block_size is not None:
        k_runtime, v_runtime, block_table = make_paged_kvcache(k_cache, v_cache, case.paged_block_size)
        cache_batch_idx = None

    def run_kvcache() -> torch.Tensor:
        return flash_attn_with_kvcache(
            q,
            k_runtime,
            v_runtime,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            block_table=block_table,
            num_splits=0,
            softmax_scale=softmax_scale,
            causal=case.causal,
        )

    def run_kvcache_nosplit() -> torch.Tensor:
        return flash_attn_with_kvcache(
            q,
            k_runtime,
            v_runtime,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            block_table=block_table,
            num_splits=1,
            softmax_scale=softmax_scale,
            causal=case.causal,
        )

    cuda_ms, wall_ms = measure(run_kvcache, warmup=warmup, repeats=repeats)
    print_row(case.name, "kvcache_api_full", cuda_ms, wall_ms)
    cuda_ms, wall_ms = measure(run_kvcache_nosplit, warmup=warmup, repeats=repeats)
    print_row(case.name, "kvcache_api_nosplit", cuda_ms, wall_ms)

    def dense_prepare() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return make_dense_inputs(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            k_new=k_new,
            v_new=v_new,
            cache_batch_idx=cache_batch_idx,
        )

    cuda_ms, wall_ms = measure(dense_prepare, warmup=warmup, repeats=repeats)
    print_row(case.name, "dense_prepare_cache", cuda_ms, wall_ms)
    _, k_used, v_used, effective_lengths = dense_prepare()

    cuda_ms, wall_ms = measure(
        lambda: make_varlen_inputs(q, k_used, v_used, effective_lengths),
        warmup=warmup,
        repeats=repeats,
    )
    print_row(case.name, "dense_fa_pack_and_metadata", cuda_ms, wall_ms)
    q_packed, k_packed, v_packed, cu_q, cu_k, max_q, max_k = make_varlen_inputs(
        q, k_used, v_used, effective_lengths
    )

    def run_varlen_attention() -> torch.Tensor:
        return flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=case.causal,
        )

    def run_dense_fa_full() -> torch.Tensor:
        local_q, local_k, local_v, local_lengths = dense_prepare()
        local_inputs = make_varlen_inputs(local_q, local_k, local_v, local_lengths)
        out = flash_attn_varlen_func(
            local_inputs[0],
            local_inputs[1],
            local_inputs[2],
            local_inputs[3],
            local_inputs[4],
            max_seqlen_q=local_inputs[5],
            max_seqlen_k=local_inputs[6],
            softmax_scale=softmax_scale,
            causal=case.causal,
        )
        return out.reshape(q.shape)

    cuda_ms, wall_ms = measure(run_varlen_attention, warmup=warmup, repeats=repeats)
    print_row(case.name, "dense_fa_attention_only", cuda_ms, wall_ms)
    cuda_ms, wall_ms = measure(run_dense_fa_full, warmup=warmup, repeats=repeats)
    print_row(case.name, "dense_fa_full_workaround", cuda_ms, wall_ms)

    cuda_ms, wall_ms = measure(
        lambda: make_sdpa_inputs(q, k_used, v_used, effective_lengths, causal=case.causal),
        warmup=warmup,
        repeats=repeats,
    )
    print_row(case.name, "padded_sdpa_materialize_and_mask", cuda_ms, wall_ms)
    q_sdpa, k_sdpa, v_sdpa, attn_mask = make_sdpa_inputs(
        q, k_used, v_used, effective_lengths, causal=case.causal
    )

    def run_sdpa_attention() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            enable_gqa=q_sdpa.shape[1] != k_sdpa.shape[1],
            scale=softmax_scale,
        )

    def run_sdpa_full() -> torch.Tensor:
        local_q, local_k, local_v, local_lengths = dense_prepare()
        sdpa_q, sdpa_k, sdpa_v, local_mask = make_sdpa_inputs(
            local_q, local_k, local_v, local_lengths, causal=case.causal
        )
        out = F.scaled_dot_product_attention(
            sdpa_q,
            sdpa_k,
            sdpa_v,
            attn_mask=local_mask,
            enable_gqa=sdpa_q.shape[1] != sdpa_k.shape[1],
            scale=softmax_scale,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    cuda_ms, wall_ms = measure(run_sdpa_attention, warmup=warmup, repeats=repeats)
    print_row(case.name, "padded_sdpa_attention_only", cuda_ms, wall_ms)
    cuda_ms, wall_ms = measure(run_sdpa_full, warmup=warmup, repeats=repeats)
    print_row(case.name, "padded_sdpa_full", cuda_ms, wall_ms)


def profile_scaling_case(
    case: BenchmarkCase,
    *,
    sweep: str,
    variable: str,
    value: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> None:
    dtype = torch.float16
    q, k_cache, v_cache, cache_seqlens, cache_batch_idx, k_new, v_new = make_case_tensors(
        case, dtype, device
    )
    softmax_scale = q.shape[-1] ** (-0.5)
    block_table = None
    k_runtime = k_cache
    v_runtime = v_cache
    if case.paged_block_size is not None:
        k_runtime, v_runtime, block_table = make_paged_kvcache(k_cache, v_cache, case.paged_block_size)
        cache_batch_idx = None

    def run_kvcache(num_splits: int) -> torch.Tensor:
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

    def dense_prepare() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return make_dense_inputs(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            k_new=k_new,
            v_new=v_new,
            cache_batch_idx=cache_batch_idx,
        )

    _, k_used, v_used, effective_lengths = dense_prepare()
    q_packed, k_packed, v_packed, cu_q, cu_k, max_q, max_k = make_varlen_inputs(
        q, k_used, v_used, effective_lengths
    )

    def run_dense_fa_attention() -> torch.Tensor:
        return flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=case.causal,
        )

    def run_dense_fa_full() -> torch.Tensor:
        local_q, local_k, local_v, local_lengths = dense_prepare()
        local_inputs = make_varlen_inputs(local_q, local_k, local_v, local_lengths)
        out = flash_attn_varlen_func(
            local_inputs[0],
            local_inputs[1],
            local_inputs[2],
            local_inputs[3],
            local_inputs[4],
            max_seqlen_q=local_inputs[5],
            max_seqlen_k=local_inputs[6],
            softmax_scale=softmax_scale,
            causal=case.causal,
        )
        return out.reshape(q.shape)

    for component, fn in [
        ("kvcache_api_full", lambda: run_kvcache(0)),
        ("kvcache_api_nosplit", lambda: run_kvcache(1)),
        ("dense_fa_attention_only", run_dense_fa_attention),
        ("dense_fa_full_workaround", run_dense_fa_full),
    ]:
        cuda_ms, wall_ms = measure(fn, warmup=warmup, repeats=repeats)
        print_scaling(sweep, case.name, variable, value, component, cuda_ms, wall_ms)


def run_scaling_sweeps(*, warmup: int, repeats: int, device: torch.device) -> None:
    for seqlen in [512, 1024, 2048, 4096, 8192, 16384]:
        for paged_block_size, layout in [(None, "contig"), (256, "paged")]:
            case = BenchmarkCase(
                f"{layout}_decode_h64_seq{seqlen}",
                2,
                2,
                1,
                seqlen,
                0,
                6,
                1,
                64,
                True,
                False,
                False,
                paged_block_size,
            )
            profile_scaling_case(
                case,
                sweep="seqlen_decode_h64",
                variable="seqlen",
                value=seqlen,
                warmup=warmup,
                repeats=repeats,
                device=device,
            )

    for batch_size in [1, 2, 4, 8, 16]:
        case = BenchmarkCase(
            f"contig_decode_h64_batch{batch_size}",
            batch_size,
            batch_size,
            1,
            4096,
            0,
            6,
            1,
            64,
            True,
            False,
            False,
        )
        profile_scaling_case(
            case,
            sweep="batch_decode_h64",
            variable="batch",
            value=batch_size,
            warmup=warmup,
            repeats=repeats,
            device=device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the report profiling suite by default: component breakdowns "
            "plus sequence-length and batch-size scaling sweeps."
        )
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--global-warmup-iters", type=int, default=80)
    parser.add_argument("--skip-breakdown", action="store_true")
    parser.add_argument("--scaling", action="store_true", help="Compatibility flag; scaling now runs by default.")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the same suites with shorter warmup/repeat counts for smoke testing.",
    )
    args = parser.parse_args()

    if args.quick:
        args.warmup = min(args.warmup, 5)
        args.repeats = min(args.repeats, 10)
        args.global_warmup_iters = min(args.global_warmup_iters, 5)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    global_gpu_warmup(device, torch.float16, iters=args.global_warmup_iters)
    print(
        f"PROFILE torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
        f"warmup={args.warmup} repeats={args.repeats}",
        flush=True,
    )

    if not args.skip_breakdown:
        cases = [
            BenchmarkCase("contig_decode_h64", 2, 2, 1, 4096, 0, 6, 1, 64, True, False, False),
            BenchmarkCase("contig_chunk_h64", 2, 2, 16, 4096, 0, 4, 2, 64, False, False, False),
            BenchmarkCase("paged_decode_h64", 2, 2, 1, 4096, 0, 6, 1, 64, True, False, False, 256),
            BenchmarkCase("paged_chunk_h64", 2, 2, 16, 4096, 0, 4, 2, 64, False, False, False, 256),
        ]
        for case in cases:
            profile_case(case, warmup=args.warmup, repeats=args.repeats, device=device)
    if not args.skip_scaling:
        run_scaling_sweeps(warmup=args.warmup, repeats=args.repeats, device=device)


if __name__ == "__main__":
    main()
