"""FlashAttention regression tests with shared helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from flash_attention_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)


# --------------------------------------------------------------------------------------
# Configuration


EXCEL_REL_EPS = 1e-6
EXCEL_TOPK_ROWS = 10_000
TEST_REL_EPS = 1e-6

SAVE_DEBUG_EXCEL = False  # flip to True to dump Excel snapshots of top errors
OUTPUT_DIR = "/outputs"

BWD_TOLS = dict(
    atol=9e-3,
    rtol=1000,
    rtol_l2=100,
    mean_atol=2e-4,
    mean_rtol=1,
    mean_rtol_l2=100,
)

DTYPES = [torch.float16]
HEAD_DIMS = [64, 128]
BATCH_SIZES = [1, 3]
SOFTMAX_SCALES = [None, 0.3]
# SOFTMAX_SCALES = [0.3]
CAUSAL_FLAGS = [False, True]
NHEAD_PAIRS = [(2, 1), (4, 2), (6, 3), (6, 1)]

SEQLEN_CASES: Sequence[Tuple[int, int]] = [
    (64, 64),
    (64, 128),
    (64, 256),
    (128, 64),
    (256, 64),
    (128, 128),
    (1024, 1024),
    (128, 256),
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (256, 128),
    (512, 128),
    (768, 128),
    (1024, 128),
    (1024, 256),
    (63, 63),
    (65, 65),
    (127, 127),
    (129, 129),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),       
    (64, 128),
    (64, 256),
    (128, 64),
    (256, 64),
    (128, 128),
    (1024, 1024),
    (128, 256),
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (256, 128),
    (512, 128),
    (768, 128),
    (1024, 128),
    (1024, 256),
    (64, 2),
    (127, 63),
    (129, 65),
    (128, 127),
    (128, 129),
    (128, 1025),
    (256, 1025),
    (128, 128),
    (1024, 1024),
    (128, 256),
    (256, 64),
    (897, 1024),
    (959, 1024),
    (960, 1024),
    (961, 1024),
    (1023, 1024),
    (1024, 1023),
    (1024, 897),
    (1,64),
    (1,128),
    (65,64),
    (65,128),
    (129,64),
    (129,128),
    (257,64),
    (257,128),
    (1, 1024),
    (1023, 1024),
    (1025, 1024),
    (64, 1),
    (128,1),
    (64, 65),
    (128,65),
    (64, 129),
    (128,129),
    (64, 257),
    (128,257),
    (1024, 1),
    (1024, 2),
    (1024, 1023),
    (1024, 1025),
]


# --------------------------------------------------------------------------------------
# Helper data structures


@dataclass
class MetricsBundle:
    output: Dict[str, float]
    dq: Dict[str, float]
    dk: Dict[str, float]
    dv: Dict[str, float]

    def items(self) -> Iterable[Tuple[str, Dict[str, float]]]:
        return (("output", self.output), ("dq", self.dq), ("dk", self.dk), ("dv", self.dv))


@dataclass
class DebugPair:
    actual: torch.Tensor
    reference: torch.Tensor


@dataclass
class VarlenTensors:
    q_packed: torch.Tensor
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    d_output_packed: torch.Tensor
    q_padded: torch.Tensor
    k_padded: torch.Tensor
    v_padded: torch.Tensor
    d_output_padded: torch.Tensor
    seqlens_q: List[int]
    seqlens_k: List[int]


# --------------------------------------------------------------------------------------
# Utility functions


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention tests")
    torch.cuda.init()
    device_index = torch.cuda.current_device()
    return torch.device(f"cuda:{device_index}")


def _tensor_with_grad(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach().requires_grad_(True)


def _error_metrics(x: torch.Tensor, ref: torch.Tensor, eps: float = TEST_REL_EPS) -> Dict[str, float]:
    diff = x - ref
    abs_err = diff.abs()
    denom = ref.abs().clamp_min(eps)
    rel_err = abs_err / denom

    diff_fp32 = diff.float()
    ref_fp32 = ref.float()
    rel_err_fp32 = rel_err.float()

    return {
        "max_abs": abs_err.max().item() if abs_err.numel() > 0 else 0.0,
        "mean_abs": abs_err.mean().item() if abs_err.numel() > 0 else 0.0,
        "max_rel": rel_err.max().item() if rel_err.numel() > 0 else 0.0,
        "mean_rel": rel_err.mean().item() if rel_err.numel() > 0 else 0.0,
        "l2_rel": (diff_fp32.norm() / (ref_fp32.norm() + eps)).item() if ref.numel() > 0 else 0.0,
        "rms_rel": rel_err_fp32.square().mean().sqrt().item() if rel_err.numel() > 0 else 0.0,
    }


def _print_metrics(bundle: MetricsBundle) -> None:
    output_metrics = bundle.output
    dq_metrics = bundle.dq
    dk_metrics = bundle.dk
    dv_metrics = bundle.dv

    print("\n")
    print("========================================")
    # print(f"seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, d = {d}, causal = {causal}")
    print(
        f"output max_abs={output_metrics['max_abs']} mean_abs={output_metrics['mean_abs']} "
        f"max_rel={output_metrics['max_rel']} mean_rel={output_metrics['mean_rel']} "
        f"l2_rel={output_metrics['l2_rel']} rms_rel={output_metrics['rms_rel']}"
    )
    print(
        f"dQ     max_abs={dq_metrics['max_abs']} mean_abs={dq_metrics['mean_abs']} "
        f"max_rel={dq_metrics['max_rel']} mean_rel={dq_metrics['mean_rel']} "
        f"l2_rel={dq_metrics['l2_rel']} rms_rel={dq_metrics['rms_rel']}"
    )
    print(
        f"dK     max_abs={dk_metrics['max_abs']} mean_abs={dk_metrics['mean_abs']} "
        f"max_rel={dk_metrics['max_rel']} mean_rel={dk_metrics['mean_rel']} "
        f"l2_rel={dk_metrics['l2_rel']} rms_rel={dk_metrics['rms_rel']}"
    )
    print(
        f"dV     max_abs={dv_metrics['max_abs']} mean_abs={dv_metrics['mean_abs']} "
        f"max_rel={dv_metrics['max_rel']} mean_rel={dv_metrics['mean_rel']} "
        f"l2_rel={dv_metrics['l2_rel']} rms_rel={dv_metrics['rms_rel']}"
    )
    print("========================================")


def _assert_metrics(bundle: MetricsBundle) -> None:
    for name, metrics in bundle.items():
        assert metrics["max_abs"] <= BWD_TOLS["atol"], f"{name} max_abs={metrics['max_abs']}"
        assert metrics["max_rel"] <= BWD_TOLS["rtol"], f"{name} max_rel={metrics['max_rel']}"
        assert metrics["l2_rel"] <= BWD_TOLS["rtol_l2"], f"{name} l2_rel={metrics['l2_rel']}"
        assert metrics["mean_abs"] <= BWD_TOLS["mean_atol"], f"{name} mean_abs={metrics['mean_abs']}"
        assert metrics["mean_rel"] <= BWD_TOLS["mean_rtol"], f"{name} mean_rel={metrics['mean_rel']}"
        assert metrics["rms_rel"] <= BWD_TOLS["mean_rtol_l2"], f"{name} rms_rel={metrics['rms_rel']}"


def _flatten_numpy(x: torch.Tensor) -> np.ndarray:
    if x.numel() == 0:
        return np.empty(0, dtype=np.float32)
    return x.detach().float().cpu().reshape(-1).numpy()


def _build_debug_tables(pairs: Dict[str, DebugPair]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name, tensors in pairs.items():
        actual = _flatten_numpy(tensors.actual)
        reference = _flatten_numpy(tensors.reference)
        diff = actual - reference
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / np.maximum(np.abs(reference), EXCEL_REL_EPS)
        base_df = pd.DataFrame(
            {
                "actual": actual,
                "reference": reference,
                "diff": diff,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )
        tables[f"{name}_abs"] = base_df.sort_values(by="abs_diff", ascending=False).head(EXCEL_TOPK_ROWS)
        tables[f"{name}_rel"] = base_df.sort_values(by="rel_diff", ascending=False).head(EXCEL_TOPK_ROWS)
    return tables


def _maybe_emit_excel(tag: str, pairs: Dict[str, DebugPair]) -> None:
    if not SAVE_DEBUG_EXCEL:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{timestamp}_{tag}.xlsx")
    tables = _build_debug_tables(pairs)

    with pd.ExcelWriter(path) as writer:
        for sheet_name, df in tables.items():
            writer_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=writer_sheet, index=False)

    print(f"Saved Excel debug file: {path}")


def causal_lower_right(seqlen_q: int, seqlen_k: int, device: torch.device) -> torch.Tensor:
    diagonal_offset = seqlen_k - seqlen_q
    return torch.tril(
        torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device),
        diagonal=diagonal_offset,
    )


def vanilla_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    d_output: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)

    nheads_q = query_torch.size(1)
    nheads_k = key_torch.size(1)
    assert nheads_q % nheads_k == 0, "nheads_q must be divisible by nheads_k"
    enable_gqa = nheads_q != nheads_k

    seqlen_q = query_torch.size(2)
    seqlen_k = key_torch.size(2)

    is_causal = False
    attn_mask = None
    if causal:
        if seqlen_q == seqlen_k:
            is_causal = True
        else:
            attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=query_torch.device)

    output_torch = F.scaled_dot_product_attention(
        query_torch,
        key_torch,
        value_torch,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
        scale=softmax_scale,
    )

    if d_output is None:
        return (output_torch.permute(0, 2, 1, 3).contiguous(),)

    d_output_torch = d_output.permute(0, 2, 1, 3).contiguous()
    d_query_torch, d_key_torch, d_value_torch = torch.autograd.grad(
        outputs=output_torch,
        inputs=(query_torch, key_torch, value_torch),
        grad_outputs=d_output_torch,
        retain_graph=False,
        allow_unused=False,
    )

    return (
        output_torch.permute(0, 2, 1, 3).contiguous(),
        d_query_torch.permute(0, 2, 1, 3).contiguous(),
        d_key_torch.permute(0, 2, 1, 3).contiguous(),
        d_value_torch.permute(0, 2, 1, 3).contiguous(),
    )


def memory_efficient_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    d_output: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return vanilla_attention_ref(query, key, value, d_output, causal=causal, softmax_scale=softmax_scale)


def _pack_padded_tensor(x: torch.Tensor, seqlens: Sequence[int]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for i, seqlen in enumerate(seqlens):
        if seqlen <= 0:
            continue
        chunks.append(x[i, :seqlen].contiguous())
    if not chunks:
        return x.new_zeros((0,) + x.shape[2:])
    return torch.cat(chunks, dim=0)


def _generate_varlen_tensors(
    *,
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> VarlenTensors:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for varlen tests")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    seqlens_q = torch.randint(1, max_seqlen_q + 1, (batch_size,), dtype=torch.int32, generator=rng)
    seqlens_k = torch.randint(1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32, generator=rng)
    seqlens_q[torch.randint(0, batch_size, (1,), generator=rng).item()] = max_seqlen_q
    seqlens_k[torch.randint(0, batch_size, (1,), generator=rng).item()] = max_seqlen_k

    cu_q = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_k = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(seqlens_k, dim=0, dtype=torch.int32)

    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())

    q_packed = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype)
    k_packed = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype)
    v_packed = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype)

    q_padded = torch.zeros(batch_size, max_seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_padded = torch.zeros(batch_size, max_seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    v_padded = torch.zeros(batch_size, max_seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)

    q_offset = 0
    for i, seqlen in enumerate(seqlens_q.tolist()):
        next_offset = q_offset + seqlen
        q_padded[i, :seqlen] = q_packed[q_offset:next_offset]
        q_offset = next_offset

    k_offset = 0
    for i, seqlen in enumerate(seqlens_k.tolist()):
        next_offset = k_offset + seqlen
        k_slice = k_packed[k_offset:next_offset]
        v_slice = v_packed[k_offset:next_offset]
        k_padded[i, :seqlen] = k_slice
        v_padded[i, :seqlen] = v_slice
        k_offset = next_offset

    d_output_padded = torch.randn(batch_size, max_seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    d_output_packed = _pack_padded_tensor(d_output_padded, [int(x) for x in seqlens_q.tolist()])

    return VarlenTensors(
        q_packed=q_packed,
        k_packed=k_packed,
        v_packed=v_packed,
        cu_seqlens_q=cu_q.to(device=device),
        cu_seqlens_k=cu_k.to(device=device),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        d_output_packed=d_output_packed,
        q_padded=q_padded,
        k_padded=k_padded,
        v_padded=v_padded,
        d_output_padded=d_output_padded,
        seqlens_q=[int(x) for x in seqlens_q.tolist()],
        seqlens_k=[int(x) for x in seqlens_k.tolist()],
    )


def _varlen_reference(
    tensors: VarlenTensors,
    *,
    causal: bool,
    softmax_scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output_ref: List[torch.Tensor] = []
    dq_ref: List[torch.Tensor] = []
    dk_ref: List[torch.Tensor] = []
    dv_ref: List[torch.Tensor] = []

    for i, (seqlen_q, seqlen_k) in enumerate(zip(tensors.seqlens_q, tensors.seqlens_k)):
        q_i = tensors.q_padded[i : i + 1, :seqlen_q]
        k_i = tensors.k_padded[i : i + 1, :seqlen_k]
        v_i = tensors.v_padded[i : i + 1, :seqlen_k]
        d_out_i = tensors.d_output_padded[i : i + 1, :seqlen_q]

        out_i, dq_i, dk_i, dv_i = vanilla_attention_ref(
            q_i,
            k_i,
            v_i,
            d_out_i,
            causal=causal,
            softmax_scale=softmax_scale,
        )

        output_ref.append(out_i.squeeze(0))
        dq_ref.append(dq_i.squeeze(0))
        dk_ref.append(dk_i.squeeze(0))
        dv_ref.append(dv_i.squeeze(0))

    return (
        torch.cat(output_ref, dim=0),
        torch.cat(dq_ref, dim=0),
        torch.cat(dk_ref, dim=0),
        torch.cat(dv_ref, dim=0),
    )


def _kvcache_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: Sequence[int],
    *,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    causal: bool,
    softmax_scale: Optional[float],
) -> torch.Tensor:
    if cache_batch_idx is not None:
        k_cache = k_cache[cache_batch_idx.to(dtype=torch.long)]
        v_cache = v_cache[cache_batch_idx.to(dtype=torch.long)]
    effective_lengths = list(cache_seqlens)
    if k_new is not None and v_new is not None:
        for batch_idx, start in enumerate(cache_seqlens):
            k_cache[batch_idx, start : start + k_new.shape[1]] = k_new[batch_idx]
            v_cache[batch_idx, start : start + v_new.shape[1]] = v_new[batch_idx]
            effective_lengths[batch_idx] = start + k_new.shape[1]
    outputs: List[torch.Tensor] = []
    for batch_idx, seqlen_k in enumerate(effective_lengths):
        out_i = vanilla_attention_ref(
            q[batch_idx : batch_idx + 1],
            k_cache[batch_idx : batch_idx + 1, :seqlen_k],
            v_cache[batch_idx : batch_idx + 1, :seqlen_k],
            causal=causal,
            softmax_scale=softmax_scale,
        )[0]
        outputs.append(out_i)
    return torch.cat(outputs, dim=0)


def _make_paged_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_block_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _materialize_paged_kvcache(
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_cache: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_num_blocks_per_seq = block_table.shape
    page_block_size = k_paged.shape[1]
    k_cache = torch.empty(
        batch_size,
        seqlen_cache,
        k_paged.shape[2],
        k_paged.shape[3],
        device=k_paged.device,
        dtype=k_paged.dtype,
    )
    v_cache = torch.empty_like(k_cache)
    for batch_idx in range(batch_size):
        for block_idx in range(max_num_blocks_per_seq):
            start = block_idx * page_block_size
            end = min(start + page_block_size, seqlen_cache)
            if start >= end:
                continue
            physical_block = int(block_table[batch_idx, block_idx].item())
            k_cache[batch_idx, start:end].copy_(k_paged[physical_block, : end - start])
            v_cache[batch_idx, start:end].copy_(v_paged[physical_block, : end - start])
    return k_cache, v_cache


def _bundle_from_tensors(
    output: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    output_ref: torch.Tensor,
    dq_ref: torch.Tensor,
    dk_ref: torch.Tensor,
    dv_ref: torch.Tensor,
) -> Tuple[MetricsBundle, Dict[str, DebugPair]]:
    bundle = MetricsBundle(
        output=_error_metrics(output, output_ref),
        dq=_error_metrics(dq, dq_ref),
        dk=_error_metrics(dk, dk_ref),
        dv=_error_metrics(dv, dv_ref),
    )
    pairs = {
        "output": DebugPair(output, output_ref),
        "dq": DebugPair(dq, dq_ref),
        "dk": DebugPair(dk, dk_ref),
        "dv": DebugPair(dv, dv_ref),
    }
    return bundle, pairs


# --------------------------------------------------------------------------------------
# Regular attention tests


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("seqlen_q, seqlen_k", SEQLEN_CASES)
def test_flash_attn(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    query = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)

    q_flash = _tensor_with_grad(query)
    k_flash = _tensor_with_grad(key)
    v_flash = _tensor_with_grad(value)

    output_flash = flash_attn_func(
        q_flash,
        k_flash,
        v_flash,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dk_flash, dv_flash = torch.autograd.grad(
        outputs=output_flash,
        inputs=(q_flash, k_flash, v_flash),
        grad_outputs=d_output.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    output_ref, dq_ref, dk_ref, dv_ref = memory_efficient_attention_ref(
        query,
        key,
        value,
        d_output,
        causal,
        softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        output_flash.detach(),
        dq_flash.detach(),
        dk_flash.detach(),
        dv_flash.detach(),
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=f"flash_attn_b{batch_size}_hq{nheads}_hk{nheads_k}_sq{seqlen_q}_sk{seqlen_k}_d{head_dim}_c{int(causal)}",
        pairs=pairs,
    )
    _assert_metrics(bundle)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("seqlen_q, seqlen_k", SEQLEN_CASES)
def test_flash_attn_kv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    query = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)

    query_flash = _tensor_with_grad(query)
    kv_flash = _tensor_with_grad(torch.stack((key, value), dim=2))

    output_flash = flash_attn_kvpacked_func(
        query_flash,
        kv_flash,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dkv_flash = torch.autograd.grad(
        outputs=output_flash,
        inputs=(query_flash, kv_flash),
        grad_outputs=d_output.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dk_flash = dkv_flash[:, :, 0].detach()
    dv_flash = dkv_flash[:, :, 1].detach()

    output_ref, dq_ref, dk_ref, dv_ref = memory_efficient_attention_ref(
        query,
        key,
        value,
        d_output,
        causal,
        softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        output_flash.detach(),
        dq_flash.detach(),
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=f"flash_attn_kv_b{batch_size}_hq{nheads}_hk{nheads_k}_sq{seqlen_q}_sk{seqlen_k}_d{head_dim}_c{int(causal)}",
        pairs=pairs,
    )
    _assert_metrics(bundle)


EQUAL_SEQLEN_CASES = [case for case in SEQLEN_CASES if case[0] == case[1]]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("seqlen", sorted({case[0] for case in EQUAL_SEQLEN_CASES}))
def test_flash_attn_qkv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    seqlen: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    query = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen, nheads_k, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen, nheads_k, head_dim, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)

    qkv_flash = _tensor_with_grad(torch.stack((query, key, value), dim=2))

    output_flash = flash_attn_qkvpacked_func(
        qkv_flash,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    (dqkv_flash,) = torch.autograd.grad(
        outputs=output_flash,
        inputs=(qkv_flash,),
        grad_outputs=d_output.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dq_flash = dqkv_flash[:, :, 0].detach()
    dk_flash = dqkv_flash[:, :, 1].detach()
    dv_flash = dqkv_flash[:, :, 2].detach()

    output_ref, dq_ref, dk_ref, dv_ref = memory_efficient_attention_ref(
        query,
        key,
        value,
        d_output,
        causal,
        softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        output_flash.detach(),
        dq_flash,
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=f"flash_attn_qkv_b{batch_size}_hq{nheads}_hk{nheads_k}_s{seqlen}_d{head_dim}_c{int(causal)}",
        pairs=pairs,
    )
    _assert_metrics(bundle)


# --------------------------------------------------------------------------------------
# Variable-length attention tests


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("max_seqlen_q, max_seqlen_k", SEQLEN_CASES)
def test_flash_attn_varlen(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    tensors = _generate_varlen_tensors(
        batch_size=batch_size,
        nheads=nheads,
        nheads_k=nheads_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    q_flash = _tensor_with_grad(tensors.q_packed)
    k_flash = _tensor_with_grad(tensors.k_packed)
    v_flash = _tensor_with_grad(tensors.v_packed)

    out_flash = flash_attn_varlen_func(
        q_flash,
        k_flash,
        v_flash,
        tensors.cu_seqlens_q,
        tensors.cu_seqlens_k,
        tensors.max_seqlen_q,
        tensors.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dk_flash, dv_flash = torch.autograd.grad(
        outputs=out_flash,
        inputs=(q_flash, k_flash, v_flash),
        grad_outputs=tensors.d_output_packed.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    output_ref, dq_ref, dk_ref, dv_ref = _varlen_reference(
        tensors,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        out_flash.detach(),
        dq_flash.detach(),
        dk_flash.detach(),
        dv_flash.detach(),
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=(
            f"flash_attn_varlen_b{batch_size}_hq{nheads}_hk{nheads_k}_"
            f"mq{max_seqlen_q}_mk{max_seqlen_k}_d{head_dim}_c{int(causal)}"
        ),
        pairs=pairs,
    )
    _assert_metrics(bundle)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("max_seqlen_q, max_seqlen_k", SEQLEN_CASES)
def test_flash_attn_varlen_kv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    tensors = _generate_varlen_tensors(
        batch_size=batch_size,
        nheads=nheads,
        nheads_k=nheads_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    q_flash = _tensor_with_grad(tensors.q_packed)
    kv_flash = _tensor_with_grad(torch.stack((tensors.k_packed, tensors.v_packed), dim=1))

    out_flash = flash_attn_varlen_kvpacked_func(
        q_flash,
        kv_flash,
        tensors.cu_seqlens_q,
        tensors.cu_seqlens_k,
        tensors.max_seqlen_q,
        tensors.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dkv_flash = torch.autograd.grad(
        outputs=out_flash,
        inputs=(q_flash, kv_flash),
        grad_outputs=tensors.d_output_packed.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dk_flash = dkv_flash[:, 0].detach()
    dv_flash = dkv_flash[:, 1].detach()

    output_ref, dq_ref, dk_ref, dv_ref = _varlen_reference(
        tensors,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        out_flash.detach(),
        dq_flash.detach(),
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=(
            f"flash_attn_varlen_kv_b{batch_size}_hq{nheads}_hk{nheads_k}_"
            f"mq{max_seqlen_q}_mk{max_seqlen_k}_d{head_dim}_c{int(causal)}"
        ),
        pairs=pairs,
    )
    _assert_metrics(bundle)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2), (6, 1)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("has_batch_idx", [False, True])
def test_flash_attn_kvcache_read(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    has_batch_idx: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    batch_size = 2
    batch_size_cache = 4 if has_batch_idx else batch_size
    seqlen_q = 4
    seqlen_cache = 17
    cache_lengths = [7, 13]

    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size_cache, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size_cache, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor(cache_lengths, device=device, dtype=torch.int32)
    cache_batch_idx = (
        torch.tensor([2, 0], device=device, dtype=torch.int32) if has_batch_idx else None
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    out_ref = _kvcache_reference(
        q,
        k_cache.clone(),
        v_cache.clone(),
        cache_lengths,
        cache_batch_idx=cache_batch_idx,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2), (6, 1)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("append_len", [1, 3])
def test_flash_attn_kvcache_append(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    has_batch_idx: bool,
    append_len: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    batch_size = 2
    seqlen_q = 4
    batch_size_cache = 4 if has_batch_idx else batch_size
    seqlen_cache = 20
    cache_lengths = [5, 9]

    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size_cache, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size_cache, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    k_new = torch.randn(batch_size, append_len, nheads_k, head_dim, device=device, dtype=dtype)
    v_new = torch.randn(batch_size, append_len, nheads_k, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor(cache_lengths, device=device, dtype=torch.int32)
    cache_batch_idx = (
        torch.tensor([3, 1], device=device, dtype=torch.int32) if has_batch_idx else None
    )

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    updated_lengths = [length + append_len for length in cache_lengths]
    for batch_idx, start in enumerate(cache_lengths):
        cache_row = cache_batch_idx[batch_idx].item() if cache_batch_idx is not None else batch_idx
        k_cache_ref[cache_row, start : start + append_len] = k_new[batch_idx]
        v_cache_ref[cache_row, start : start + append_len] = v_new[batch_idx]

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    out_ref = _kvcache_reference(
        q,
        k_cache.clone(),
        v_cache.clone(),
        cache_lengths,
        k_new=k_new,
        v_new=v_new,
        cache_batch_idx=cache_batch_idx,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_cache, k_cache_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v_cache, v_cache_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2), (6, 1)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("num_splits", [0, 1, 2, 3])
def test_flash_attn_kvcache_splitkv(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    causal: bool,
    num_splits: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    batch_size = 2
    seqlen_q = 4
    seqlen_cache = 31
    cache_lengths = [17, 29]

    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor(cache_lengths, device=device, dtype=torch.int32)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        causal=causal,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()

    out_ref = _kvcache_reference(
        q,
        k_cache.clone(),
        v_cache.clone(),
        cache_lengths,
        causal=causal,
        softmax_scale=None,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2), (6, 1)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("num_splits", [0, 2])
def test_flash_attn_kvcache_paged_read(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    causal: bool,
    num_splits: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    batch_size = 2
    seqlen_q = 4
    seqlen_cache = 300
    cache_lengths = [123, 257]

    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    k_paged, v_paged, block_table = _make_paged_kvcache(k_dense, v_dense)
    cache_seqlens = torch.tensor(cache_lengths, device=device, dtype=torch.int32)

    out = flash_attn_with_kvcache(
        q,
        k_paged,
        v_paged,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=causal,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()

    out_ref = _kvcache_reference(
        q,
        k_dense,
        v_dense,
        cache_lengths,
        causal=causal,
        softmax_scale=None,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
def test_flash_attn_kvcache_paged_append(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    batch_size = 2
    seqlen_q = 4
    seqlen_cache = 300
    append_len = 3
    cache_lengths = [123, 250]

    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    k_new = torch.randn(batch_size, append_len, nheads_k, head_dim, device=device, dtype=dtype)
    v_new = torch.randn(batch_size, append_len, nheads_k, head_dim, device=device, dtype=dtype)
    k_paged, v_paged, block_table = _make_paged_kvcache(k_dense, v_dense)
    cache_seqlens = torch.tensor(cache_lengths, device=device, dtype=torch.int32)

    k_dense_ref = k_dense.clone()
    v_dense_ref = v_dense.clone()
    for batch_idx, start in enumerate(cache_lengths):
        k_dense_ref[batch_idx, start : start + append_len] = k_new[batch_idx]
        v_dense_ref[batch_idx, start : start + append_len] = v_new[batch_idx]

    out = flash_attn_with_kvcache(
        q,
        k_paged,
        v_paged,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=causal,
        num_splits=2,
    )
    torch.cuda.synchronize()

    out_ref = _kvcache_reference(
        q,
        k_dense.clone(),
        v_dense.clone(),
        cache_lengths,
        k_new=k_new,
        v_new=v_new,
        causal=causal,
        softmax_scale=None,
    )
    k_dense_after, v_dense_after = _materialize_paged_kvcache(
        k_paged, v_paged, block_table, seqlen_cache
    )
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_dense_after, k_dense_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v_dense_after, v_dense_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2), (6, 1)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("num_splits", [0, 1, 2, 4, 8])
def test_flash_attn_kvcache_splitkv_randomized_lengths(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    causal: bool,
    num_splits: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    batch_size = 3
    seqlen_q = 7
    seqlen_cache = 321
    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)

    for _ in range(8):
        cache_lengths = torch.randint(
            low=32,
            high=seqlen_cache - 1,
            size=(batch_size,),
            device=device,
            dtype=torch.int32,
        )
        out = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_lengths,
            causal=causal,
            num_splits=num_splits,
        )
        out_ref = _kvcache_reference(
            q,
            k_cache.clone(),
            v_cache.clone(),
            cache_lengths.detach().cpu().tolist(),
            causal=causal,
            softmax_scale=None,
        )
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 2)])
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("num_splits", [0, 2, 4])
@pytest.mark.parametrize("page_block_size", [256, 512])
def test_flash_attn_kvcache_paged_randomized_lengths(
    nheads: int,
    nheads_k: int,
    head_dim: int,
    causal: bool,
    num_splits: int,
    page_block_size: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    batch_size = 2
    seqlen_q = 5
    seqlen_cache = page_block_size * 3
    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    k_paged, v_paged, block_table = _make_paged_kvcache(
        k_dense, v_dense, page_block_size=page_block_size
    )

    for _ in range(6):
        cache_lengths = torch.randint(
            low=page_block_size // 2,
            high=seqlen_cache - 1,
            size=(batch_size,),
            device=device,
            dtype=torch.int32,
        )
        out = flash_attn_with_kvcache(
            q,
            k_paged.clone(),
            v_paged.clone(),
            cache_seqlens=cache_lengths,
            block_table=block_table,
            causal=causal,
            num_splits=num_splits,
        )
        out_ref = _kvcache_reference(
            q,
            k_dense.clone(),
            v_dense.clone(),
            cache_lengths.detach().cpu().tolist(),
            causal=causal,
            softmax_scale=None,
        )
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flash_attn_kvcache_paged_decode_long_context_no_error(
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    batch_size = 2
    seqlen_q = 1
    seqlen_cache = 16384
    nheads = 6
    nheads_k = 1
    page_block_size = 512
    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    v_dense = torch.randn(batch_size, seqlen_cache, nheads_k, head_dim, device=device, dtype=dtype)
    k_paged, v_paged, block_table = _make_paged_kvcache(
        k_dense, v_dense, page_block_size=page_block_size
    )
    cache_lengths = torch.full((batch_size,), seqlen_cache - 1, device=device, dtype=torch.int32)
    out = flash_attn_with_kvcache(
        q,
        k_paged,
        v_paged,
        cache_seqlens=cache_lengths,
        block_table=block_table,
        causal=True,
        num_splits=0,  # heuristic path should remain launch-safe at long context.
    )
    out_ref = _kvcache_reference(
        q,
        k_dense.clone(),
        v_dense.clone(),
        cache_lengths.detach().cpu().tolist(),
        causal=True,
        softmax_scale=None,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


def test_flash_attn_kvcache_requires_cache_seqlens_for_append() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(1, 8, 2, 64)
    v_cache = torch.randn(1, 8, 2, 64)
    k = torch.randn(1, 1, 2, 64)
    v = torch.randn(1, 1, 2, 64)

    with pytest.raises(ValueError, match="cache_seqlens is required when appending"):
        flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v)


def test_flash_attn_kvcache_requires_paired_kv() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(1, 8, 2, 64)
    v_cache = torch.randn(1, 8, 2, 64)
    k = torch.randn(1, 2, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int32)

    with pytest.raises(ValueError, match="k and v must either both be provided"):
        flash_attn_with_kvcache(q, k_cache, v_cache, k=k, cache_seqlens=cache_seqlens)


def test_flash_attn_kvcache_requires_int32_cache_seqlens() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(1, 8, 2, 64)
    v_cache = torch.randn(1, 8, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int64)

    with pytest.raises(TypeError, match="cache_seqlens must be a torch.int32 tensor"):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)


def test_flash_attn_kvcache_rejects_cpu_tensors() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(1, 8, 2, 64)
    v_cache = torch.randn(1, 8, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)


def test_flash_attn_kvcache_requires_batch_sized_cache_seqlens() -> None:
    q = torch.randn(2, 2, 4, 64)
    k_cache = torch.randn(2, 8, 2, 64)
    v_cache = torch.randn(2, 8, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int32)

    with pytest.raises(ValueError, match=r"cache_seqlens must have shape \[batch_size\]"):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)


def test_flash_attn_kvcache_accepts_shorter_append_length() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
    q_cuda = q.to(device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 2, 64)
    v = torch.randn(1, 1, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int32, device="cuda")

    out = flash_attn_with_kvcache(
        q_cuda,
        k_cache,
        v_cache,
        k=k.to(device="cuda", dtype=torch.float16),
        v=v.to(device="cuda", dtype=torch.float16),
        cache_seqlens=cache_seqlens,
    )
    assert out.shape == q_cuda.shape


def test_flash_attn_kvcache_requires_int32_cache_batch_idx() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(2, 8, 2, 64)
    v_cache = torch.randn(2, 8, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int32)
    cache_batch_idx = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(TypeError, match="cache_batch_idx must be a torch.int32 tensor"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
        )


def test_flash_attn_kvcache_requires_batch_sized_cache_batch_idx() -> None:
    q = torch.randn(2, 2, 4, 64)
    k_cache = torch.randn(3, 8, 2, 64)
    v_cache = torch.randn(3, 8, 2, 64)
    cache_seqlens = torch.tensor([4, 5], dtype=torch.int32)
    cache_batch_idx = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match=r"cache_batch_idx must have shape \[batch_size\]"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
        )


def test_flash_attn_kvcache_rejects_capacity_overrun() -> None:
    q = torch.randn(1, 2, 4, 64)
    k_cache = torch.randn(1, 5, 2, 64)
    v_cache = torch.randn(1, 5, 2, 64)
    k = torch.randn(1, 2, 2, 64)
    v = torch.randn(1, 2, 2, 64)
    cache_seqlens = torch.tensor([4], dtype=torch.int32)

    with pytest.raises(ValueError, match="cache capacity is insufficient"):
        flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("max_seqlen", sorted({case[0] for case in SEQLEN_CASES if case[0] == case[1]}))
def test_flash_attn_varlen_qkv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    tensors = _generate_varlen_tensors(
        batch_size=batch_size,
        nheads=nheads,
        nheads_k=nheads_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    qkv_packed = torch.stack((tensors.q_packed, tensors.k_packed, tensors.v_packed), dim=1)
    qkv_flash = _tensor_with_grad(qkv_packed)

    out_flash = flash_attn_varlen_qkvpacked_func(
        qkv_flash,
        tensors.cu_seqlens_q,
        tensors.max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    (dqkv_flash,) = torch.autograd.grad(
        outputs=out_flash,
        inputs=(qkv_flash,),
        grad_outputs=tensors.d_output_packed.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dq_flash = dqkv_flash[:, 0].detach()
    dk_flash = dqkv_flash[:, 1].detach()
    dv_flash = dqkv_flash[:, 2].detach()

    output_ref, dq_ref, dk_ref, dv_ref = _varlen_reference(
        tensors,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        out_flash.detach(),
        dq_flash,
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=(
            f"flash_attn_varlen_qkv_b{batch_size}_hq{nheads}_hk{nheads_k}_"
            f"m{max_seqlen}_d{head_dim}_c{int(causal)}"
        ),
        pairs=pairs,
    )
    _assert_metrics(bundle)
