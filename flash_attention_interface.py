"""Python interface for FlashAttention Turing extension."""

from typing import Optional, Tuple

import flash_attn_turing as flash_attn_gpu
import torch


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.fwd(
        q,
        k,
        v,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.varlen_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


def _flash_attn_kvcache_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cache_seqlens: torch.Tensor,
    cache_batch_idx: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    num_splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k_cache, v_cache, k, v, cache_seqlens, cache_batch_idx, block_table = [
        maybe_contiguous(x)
        for x in (q, k_cache, v_cache, k, v, cache_seqlens, cache_batch_idx, block_table)
    ]
    try:
        out, lse = flash_attn_gpu.fwd_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            None,
            None,
            cache_batch_idx,
            None,
            block_table,
            None,
            None,
            softmax_scale,
            causal,
            -1,
            -1,
            0.0,
            True,
            num_splits,
        )
    except TypeError:
        if block_table is not None or num_splits not in (0, 1):
            raise
        # Backward-compatible call path for previously built extensions that
        # only expose the legacy 9-argument KV-cache binding.
        out, lse = flash_attn_gpu.fwd_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            cache_batch_idx,
            softmax_scale,
            causal,
        )
    return out, lse


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=2, index=0).contiguous(),
            qkv.select(dim=2, index=1).contiguous(),
            qkv.select(dim=2, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, :, 0] = dq
        dqkv[:, :, 1] = dk
        dqkv[:, :, 2] = dv
        return dqkv, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=2, index=0).contiguous(),
            kv.select(dim=2, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, :, 0] = dk
        dkv[:, :, 1] = dv
        return dq, dkv, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=1, index=0).contiguous(),
            qkv.select(dim=1, index=1).contiguous(),
            qkv.select(dim=1, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.softmax_scale,
            ctx.causal,
        )
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, 0] = dq
        dqkv[:, 1] = dk
        dqkv[:, 2] = dv
        return dqkv, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=1, index=0).contiguous(),
            kv.select(dim=1, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, 0] = dk
        dkv[:, 1] = dv
        return dq, dkv, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total, nheads, headdim).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def _validate_unsupported_kvcache_args(
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    window_size: Tuple[int, int],
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("Turing KV-cache does not support rotary embeddings yet")
    if cache_leftpad is not None:
        raise NotImplementedError("Turing KV-cache does not support cache_leftpad yet")
    if window_size != (-1, -1):
        raise NotImplementedError("Turing KV-cache does not support local window attention yet")
    if softcap != 0.0:
        raise NotImplementedError("Turing KV-cache does not support softcap yet")
    if alibi_slopes is not None:
        raise NotImplementedError("Turing KV-cache does not support ALiBi yet")


def _copy_kv_to_contiguous_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cache_seqlens: torch.Tensor,
    cache_batch_idx: Optional[torch.Tensor],
) -> None:
    if k is None or v is None:
        return
    cache_lengths = cache_seqlens.detach().cpu().tolist()
    cache_rows = (
        cache_batch_idx.detach().cpu().tolist()
        if cache_batch_idx is not None
        else list(range(k.shape[0]))
    )
    for batch_idx, start in enumerate(cache_lengths):
        cache_row = cache_rows[batch_idx]
        end = start + k.shape[1]
        k_cache[cache_row, start:end].copy_(k[batch_idx])
        v_cache[cache_row, start:end].copy_(v[batch_idx])


def _copy_kv_to_paged_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    if k is None or v is None:
        return
    page_block_size = k_cache.shape[1]
    cache_lengths = cache_seqlens.detach().cpu().tolist()
    pages = block_table.detach().cpu()
    for batch_idx, start in enumerate(cache_lengths):
        for token_idx in range(k.shape[1]):
            logical_pos = start + token_idx
            table_idx = logical_pos // page_block_size
            page_offset = logical_pos % page_block_size
            physical_block = int(pages[batch_idx, table_idx].item())
            k_cache[physical_block, page_offset].copy_(k[batch_idx, token_idx])
            v_cache[physical_block, page_offset].copy_(v[batch_idx, token_idx])


def _materialize_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_seqlen = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0
    if block_table is None:
        k_used = k_cache if cache_batch_idx is None else k_cache[cache_batch_idx.to(dtype=torch.long)]
        v_used = v_cache if cache_batch_idx is None else v_cache[cache_batch_idx.to(dtype=torch.long)]
        return k_used[:, :max_seqlen], v_used[:, :max_seqlen]

    page_block_size = k_cache.shape[1]
    batch_size = block_table.shape[0]
    k_used = torch.empty(
        (batch_size, max_seqlen, k_cache.shape[2], k_cache.shape[3]),
        device=k_cache.device,
        dtype=k_cache.dtype,
    )
    v_used = torch.empty_like(k_used)
    cache_lengths = cache_seqlens.detach().cpu().tolist()
    pages = block_table.detach().cpu()
    for batch_idx, seqlen in enumerate(cache_lengths):
        pos = 0
        while pos < seqlen:
            table_idx = pos // page_block_size
            page_offset = pos % page_block_size
            physical_block = int(pages[batch_idx, table_idx].item())
            take = min(page_block_size - page_offset, seqlen - pos)
            k_used[batch_idx, pos : pos + take].copy_(
                k_cache[physical_block, page_offset : page_offset + take]
            )
            v_used[batch_idx, pos : pos + take].copy_(
                v_cache[physical_block, page_offset : page_offset + take]
            )
            pos += take
        if seqlen < max_seqlen:
            k_used[batch_idx, seqlen:].zero_()
            v_used[batch_idx, seqlen:].zero_()
    return k_used, v_used


def _attention_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    num_splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seqlen_q, nheads, head_dim = q.shape
    nheads_k = k.shape[2]
    groups = nheads // nheads_k
    cache_lengths = cache_seqlens.detach().cpu().tolist()
    out = torch.empty_like(q)
    lse = torch.empty((batch_size, nheads, seqlen_q), device=q.device, dtype=torch.float32)

    for batch_idx, seqlen_k in enumerate(cache_lengths):
        qi = q[batch_idx].float().transpose(0, 1)
        ki = k[batch_idx, :seqlen_k].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        vi = v[batch_idx, :seqlen_k].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        split_count = max(1, min(num_splits, seqlen_k)) if num_splits > 0 else 1
        partial_out = []
        partial_lse = []
        for split_idx in range(split_count):
            start = (seqlen_k * split_idx) // split_count
            end = (seqlen_k * (split_idx + 1)) // split_count
            scores = torch.matmul(qi, ki[:, start:end].transpose(-1, -2)) * softmax_scale
            if causal:
                q_pos = torch.arange(seqlen_q, device=q.device)[:, None]
                k_pos = torch.arange(start, end, device=q.device)[None, :]
                keep = k_pos <= q_pos + seqlen_k - seqlen_q
                scores = scores.masked_fill(~keep, float("-inf"))
            row_lse = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            partial_out.append(torch.matmul(probs, vi[:, start:end]))
            partial_lse.append(row_lse)

        lse_stack = torch.stack(partial_lse, dim=0)
        lse_i = torch.logsumexp(lse_stack, dim=0)
        weights = torch.exp(lse_stack - lse_i.unsqueeze(0))
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        out_i = sum(weights[idx].unsqueeze(-1) * partial_out[idx] for idx in range(split_count))
        out[batch_idx].copy_(out_i.transpose(0, 1).to(dtype=q.dtype))
        lse[batch_idx].copy_(lse_i)
    return out, lse


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor | int] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    KV-cache attention for inference on SM75.

    The signature matches upstream FlashAttention. Turing currently supports contiguous cache,
    paged cache via block_table, and split-KV. Rotary, ALiBi, local window attention, softcap,
    and cache_leftpad are not implemented for this architecture path.
    """
    _validate_unsupported_kvcache_args(
        rotary_cos, rotary_sin, cache_leftpad, window_size, softcap, alibi_slopes
    )
    if (k is None) != (v is None):
        raise ValueError("k and v must either both be provided or both be None")
    if cache_seqlens is None:
        if k is not None:
            raise ValueError("cache_seqlens is required when appending k and v")
        max_len = k_cache.shape[1] if block_table is None else block_table.shape[1] * k_cache.shape[1]
        cache_seqlens = torch.full((q.shape[0],), max_len, dtype=torch.int32, device=k_cache.device)
    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
    elif not isinstance(cache_seqlens, torch.Tensor):
        raise TypeError("cache_seqlens must be an int or a torch.Tensor")
    elif cache_seqlens.dtype != torch.int32:
        raise TypeError("cache_seqlens must be a torch.int32 tensor")
    if cache_seqlens.dim() != 1 or cache_seqlens.numel() != q.shape[0]:
        raise ValueError("cache_seqlens must have shape [batch_size]")
    if block_table is not None:
        if not isinstance(block_table, torch.Tensor):
            raise TypeError("block_table must be a torch.Tensor")
        if block_table.dtype != torch.int32:
            raise TypeError("block_table must be a torch.int32 tensor")
        if block_table.dim() != 2 or block_table.shape[0] != q.shape[0]:
            raise ValueError("block_table must have shape [batch_size, max_num_blocks_per_seq]")
        if cache_batch_idx is not None:
            raise NotImplementedError("Paged KV cache does not support cache_batch_idx")
        if k_cache.shape[1] % 256 != 0:
            raise ValueError("Paged KV cache block size must be divisible by 256")
    if cache_batch_idx is not None:
        if not isinstance(cache_batch_idx, torch.Tensor):
            raise TypeError("cache_batch_idx must be a torch.Tensor")
        if cache_batch_idx.dtype != torch.int32:
            raise TypeError("cache_batch_idx must be a torch.int32 tensor")
        if cache_batch_idx.dim() != 1 or cache_batch_idx.numel() != q.shape[0]:
            raise ValueError("cache_batch_idx must have shape [batch_size]")
    if num_splits < 0:
        raise ValueError("num_splits must be non-negative")
    seqlen_new = 0
    if k is not None and v is not None:
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            raise ValueError("k and v batch size must match q")
        if k.shape[1] != v.shape[1]:
            raise ValueError("k and v sequence length must match")
        seqlen_new = k.shape[1]
    max_cache_usage = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0
    cache_capacity = k_cache.shape[1] if block_table is None else block_table.shape[1] * k_cache.shape[1]
    if max_cache_usage + seqlen_new > cache_capacity:
        raise ValueError("cache capacity is insufficient for the requested append length")
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise RuntimeError("q, k_cache, v_cache must be CUDA tensors")
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        raise RuntimeError("q, k_cache, v_cache must be float16 tensors")
    if k is not None and v is not None:
        if not (k.is_cuda and v.is_cuda):
            raise RuntimeError("k and v must be CUDA tensors")
        if k.dtype != torch.float16 or v.dtype != torch.float16:
            raise RuntimeError("k and v must be float16 tensors")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    q, k_cache, v_cache, k, v, cache_seqlens, cache_batch_idx, block_table = [
        maybe_contiguous(x)
        for x in (q, k_cache, v_cache, k, v, cache_seqlens, cache_batch_idx, block_table)
    ]
    if block_table is not None or num_splits != 1:
        cache_seqlens_end = cache_seqlens + seqlen_new
        try:
            out, lse = _flash_attn_kvcache_forward(
                q,
                k_cache,
                v_cache,
                k,
                v,
                cache_seqlens,
                cache_batch_idx,
                block_table,
                softmax_scale,
                causal,
                num_splits,
            )
        except TypeError:
            if block_table is None:
                _copy_kv_to_contiguous_cache(k_cache, v_cache, k, v, cache_seqlens, cache_batch_idx)
            else:
                _copy_kv_to_paged_cache(k_cache, v_cache, k, v, cache_seqlens, block_table)
            k_used, v_used = _materialize_kv_cache(
                k_cache, v_cache, cache_seqlens_end, cache_batch_idx, block_table
            )
            split_count = num_splits
            if split_count == 0:
                block_n = 256 if q.shape[-1] <= 64 else 128
                split_count = max(
                    1,
                    min(128, (int(cache_seqlens_end.max().item()) + block_n - 1) // block_n),
                )
            out, lse = _attention_forward_ref(
                q, k_used, v_used, cache_seqlens_end, softmax_scale, causal, split_count
            )
        return (out, lse) if return_softmax_lse else out
    out, lse = _flash_attn_kvcache_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        cache_batch_idx,
        block_table,
        softmax_scale,
        causal,
        num_splits,
    )
    return (out, lse) if return_softmax_lse else out
