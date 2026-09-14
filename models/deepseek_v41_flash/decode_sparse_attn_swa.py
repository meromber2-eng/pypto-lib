# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DeepSeek-V4.1-Flash sliding-window attention for decode.

The V4.1 reference keeps one 128-row ring per request.  ``window_topk`` is
the materialized result of the reference ``get_window_topk_idxs`` helper:
rows are in oldest-to-newest ring order during decode and ``-1`` marks an
unwritten slot.  The PTO kernel gathers those rows, performs causal softmax
with the learned attention sink, removes query RoPE from the value output,
and applies the grouped low-rank output projection.

The cache is BF16 at this operator boundary, matching the reference's in-place
FP8 quantize/dequantize result.  The caller supplies post-RoPE KV rows after
that quantization step, so the kernel preserves the reference ordering,
masking, and attention arithmetic.
"""

from functools import lru_cache

import pypto.language as pl

from config import FLASH as M
from config import BLOCK_SIZE, DECODE_BATCH, DECODE_ORI_BLOCK_NUM, DECODE_SEQ


# Dynamic token extent is useful for a padded serving batch while all model
# dimensions remain those published in config.json.
T_DYN = pl.dynamic("T_DYN")

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
SOFTMAX_SCALE = M.softmax_scale
NEG_INF = -1.0e20

H_TILE = 16
QK_K_TILE = 128
PROJ_A_N_TILE = 128
PROJ_K_TILE = 256
PROJ_D_TILE = 256
TOPK = WIN
SPARSE_BLOCKS = 1
PADDED_TOPK = WIN
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM

assert WIN == QK_K_TILE
assert H % H_TILE == 0
assert D % PROJ_D_TILE == 0
assert O_GROUP_IN % PROJ_K_TILE == 0
assert O_LORA % PROJ_A_N_TILE == 0
assert BLOCK_SIZE == WIN


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Return the same ring indices as V4.1's Hugging Face reference.

    The helper intentionally imports torch lazily so the PTO module remains
    importable in environments that only compile kernels.  The PTO entry uses
    a fixed ``[T, WIN]`` contract; callers pad the shorter prefill result with
    ``-1`` before uploading it.
    """
    import torch

    if window_size <= 0 or bsz <= 0 or seqlen <= 0 or start_pos < 0:
        raise ValueError("window_size, bsz, and seqlen must be positive; start_pos must be non-negative")
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.to(torch.int32).unsqueeze(0).expand(bsz, -1, -1).contiguous()


def prepare_window_kv(window_kv_cache, kv, start_pos: int):
    """Apply the reference prefill/decode ring-cache update in Python.

    ``kv`` is ``[batch, sequence, head_dim]`` and the returned window is the
    source tensor consumed by the attention call.  The cache is updated
    in-place, exactly as the reference implementation does.
    """
    bsz, seqlen, _ = kv.shape
    win = window_kv_cache.shape[1]
    if start_pos == 0:
        if seqlen <= win:
            window_kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            tail = kv[:, -win:]
            window_kv_cache[:bsz, cutoff:win] = tail[:, : win - cutoff]
            window_kv_cache[:bsz, :cutoff] = tail[:, win - cutoff :]
        return kv
    if seqlen != 1:
        raise ValueError("decode SWA expects one KV row per request")
    window_kv_cache[:bsz, start_pos % win] = kv[:, 0]
    return window_kv_cache[:bsz]


def pad_window_topk(topk, window_size: int = WIN):
    """Pad the variable-width HF prefill index tensor to the PTO window width."""
    import torch

    if topk.ndim != 3 or topk.shape[-1] > window_size:
        raise ValueError("topk must have shape [batch, sequence, width <= window_size]")
    if topk.shape[-1] == window_size:
        return topk.to(torch.int32).contiguous()
    pad = torch.full(
        (*topk.shape[:-1], window_size - topk.shape[-1]),
        -1,
        dtype=torch.int32,
        device=topk.device,
    )
    return torch.cat([topk.to(torch.int32), pad], dim=-1).contiguous()


def build_window_metadata(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Build the fixed-width PTO metadata from the variable-width HF rows."""
    topk = get_window_topk_idxs(window_size, bsz, seqlen, start_pos)
    padded = pad_window_topk(topk, window_size)
    lens = (padded >= 0).sum(dim=-1).to(padded.device, dtype=padded.dtype)
    return padded.reshape(bsz * seqlen, window_size), lens.reshape(bsz * seqlen)


def golden_sparse_attn(tensors):
    """Torch reference for the PTO entry, including the sink and inverse RoPE."""
    import torch

    q = tensors["q"].float()
    cache = tensors["window_kv_cache"].float()
    topk = tensors["window_topk"].to(torch.int64)
    sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b = tensors["wo_b"].float()
    token_dim = q.shape[0]
    # This standalone entry is the one-token decode path (S == 1).  Use the
    # contract directly so a partially populated dynamic batch still maps each
    # flattened token to its request instead of dividing by the fixed capacity.
    seq = S
    heads = q.shape[1]
    out_heads = torch.zeros_like(q)
    for t in range(token_dim):
        b = t // seq
        slots = topk[t]
        valid = slots >= 0
        if not valid.any():
            continue
        kv = cache[b, slots[valid]]
        scores = q[t] @ kv.transpose(0, 1) * SOFTMAX_SCALE
        mi = scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(scores - mi)
        numerator = weights.to(torch.bfloat16).float() @ kv.to(torch.bfloat16).float()
        denominator = weights.sum(dim=-1, keepdim=True) + torch.exp(sink[:, None] - mi)
        value = numerator / denominator
        tail = value[:, NOPE_DIM:]
        even, odd = tail[..., 0::2], tail[..., 1::2]
        c = cos[t, :HALF_ROPE]
        s = sin[t, :HALF_ROPE]
        inverse = torch.empty_like(tail)
        inverse[..., 0::2] = even * c + odd * s
        inverse[..., 1::2] = odd * c - even * s
        out_heads[t] = torch.cat([value[:, :NOPE_DIM], inverse], dim=-1).to(torch.bfloat16).float()

    grouped = out_heads.view(token_dim, O_GROUPS, O_GROUP_IN)
    low_rank = torch.einsum("tgd,grd->tgr", grouped, wo_a).to(torch.bfloat16).float()
    result = torch.einsum("tk,dk->td", low_rank.reshape(token_dim, -1), wo_b)
    tensors["attn_out"][:] = result.to(torch.bfloat16)
    return tensors["attn_out"]


@pl.jit.inline
def write_decode_window_kv(
    window_kv_cache: pl.Tensor[[B, WIN, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
):
    """Write one decoded KV row per request into the V4.1 ring slot."""
    token_dim = pl.tensor.dim(kv, 0)
    cache_flat = pl.reshape(window_kv_cache, [B * WIN, HEAD_DIM])
    with pl.spmd(B, name_hint="swa_ring_write"):
        b = pl.tile.get_block_idx()
        if b < token_dim:
            slot = pl.cast(start_pos % WIN, pl.INDEX)
            dst = b * WIN + slot
            cache_flat[dst : dst + 1, 0:HEAD_DIM] = kv[b : b + 1, 0:HEAD_DIM]
    return window_kv_cache


@pl.jit.inline
def gather_decode_window(
    window_kv_cache: pl.Tensor[[B, WIN, HEAD_DIM], pl.BF16],
    window_topk: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_lens: pl.Tensor[[T_DYN], pl.INT32],
    staged_kv: pl.Tensor[[T_DYN, WIN, HEAD_DIM], pl.BF16],
):
    """Gather each request's ring slots into one contiguous per-query tile.

    ``window_topk`` is authoritative because the HF ring can put valid slots at
    the end of a partially filled row.  ``window_lens`` remains in the ABI for
    callers that share metadata with the prefill path.
    """
    token_dim = pl.tensor.dim(window_topk, 0)
    cache_flat = pl.reshape(window_kv_cache, [B * WIN, HEAD_DIM])
    staged_flat = pl.reshape(staged_kv, [token_dim * WIN, HEAD_DIM])
    with pl.spmd(token_dim, name_hint="swa_ring_gather"):
        t = pl.tile.get_block_idx()
        b = t // S
        base = t * WIN
        for w in pl.range(WIN):
            slot = pl.read(window_topk, [t, w])
            if slot >= 0:
                src = b * WIN + pl.cast(slot, pl.INDEX)
                staged_flat[base + w : base + w + 1, 0:HEAD_DIM] = cache_flat[src : src + 1, 0:HEAD_DIM]
            else:
                staged_flat[base + w : base + w + 1, 0:HEAD_DIM] = pl.full(
                    [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                )
    return staged_kv


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    window_kv_cache: pl.Tensor[[B, WIN, HEAD_DIM], pl.BF16],
    window_topk: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_lens: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Run V4.1 SWA attention, inverse RoPE, and grouped output projection."""
    token_dim = pl.tensor.dim(q, 0)
    staged_kv = pl.create_tensor([token_dim, WIN, HEAD_DIM], dtype=pl.BF16)
    staged_kv = gather_decode_window(window_kv_cache, window_topk, window_lens, staged_kv)

    q_flat = pl.reshape(q, [token_dim * H, HEAD_DIM])
    staged_kv_flat = pl.reshape(staged_kv, [token_dim * WIN, HEAD_DIM])
    o_grouped = pl.create_tensor([token_dim * O_GROUPS, O_GROUP_IN], dtype=pl.BF16)
    with pl.spmd(token_dim, name_hint="swa_qk_pv") as qk_tid:
        t = pl.tile.get_block_idx()
        kv_base = t * WIN
        kv_tile = staged_kv_flat[kv_base : kv_base + WIN, 0:HEAD_DIM]
        # Valid ring slots are not a prefix while the ring is filling.
        slots = pl.cast(window_topk[t : t + 1, 0:WIN], target_type=pl.FP32)
        slot_valid = pl.minimum(pl.maximum(pl.add(slots, 1.0), 0.0), 1.0)
        valid = pl.col_expand_mul(pl.full([H_TILE, WIN], dtype=pl.FP32, value=1.0), slot_valid)
        bias = pl.mul(pl.sub(valid, 1.0), -NEG_INF)

        for hb in pl.range(H // H_TILE):
            h0 = hb * H_TILE
            q_tile = q_flat[t * H + h0 : t * H + h0 + H_TILE, 0:HEAD_DIM]
            scores = pl.mul(pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32), SOFTMAX_SCALE)
            scores = pl.add(scores, bias)
            mi = pl.row_max(scores)
            exp_scores = pl.exp(pl.row_expand_sub(scores, mi))
            li = pl.row_sum(exp_scores)
            oi = pl.matmul(pl.cast(exp_scores, target_type=pl.BF16), kv_tile, out_dtype=pl.FP32)
            sink = pl.reshape(attn_sink[h0 : h0 + H_TILE], [H_TILE, 1])
            denom = pl.add(li, pl.exp(pl.sub(sink, mi)))
            normalized = pl.row_expand_div(oi, denom)

            normalized_nope = pl.cast(normalized[:, 0:NOPE_DIM], target_type=pl.BF16, mode="rint")
            rope = normalized[:, NOPE_DIM:HEAD_DIM]
            even = pl.gather(rope, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(rope, mask_pattern=pl.tile.MaskPattern.P1010)
            cos = pl.cast(freqs_cos[t : t + 1, 0:HALF_ROPE], target_type=pl.FP32)
            sin = pl.cast(freqs_sin[t : t + 1, 0:HALF_ROPE], target_type=pl.FP32)
            inv_even = pl.add(pl.col_expand_mul(even, cos), pl.col_expand_mul(odd, sin))
            inv_odd = pl.sub(pl.col_expand_mul(odd, cos), pl.col_expand_mul(even, sin))
            rotated = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(inv_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(inv_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")
            for group_offset in pl.unroll(H_TILE // HEADS_PER_GROUP):
                group_h0 = group_offset * HEADS_PER_GROUP
                group = pl.concat(
                    normalized_nope[group_h0 : group_h0 + HEADS_PER_GROUP, 0:NOPE_DIM],
                    rotated_bf16[group_h0 : group_h0 + HEADS_PER_GROUP, 0:ROPE_DIM],
                )
                group_id = h0 // HEADS_PER_GROUP + group_offset
                dst = t * O_GROUPS + group_id
                o_grouped[dst : dst + 1, 0:O_GROUP_IN] = pl.reshape(group, [1, O_GROUP_IN])

    # Grouped wo_a is block diagonal over heads.  Materialize the low-rank
    # groups first, then run the shared wo_b projection in D tiles.
    o_lora_flat = pl.create_tensor([token_dim * O_GROUPS, O_LORA], dtype=pl.BF16)
    wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_LORA, O_GROUP_IN])
    with pl.spmd(token_dim * O_GROUPS, name_hint="swa_o_proj_a", deps=[qk_tid]) as pa_tid:
        idx = pl.tile.get_block_idx()
        t = idx // O_GROUPS
        g = idx - t * O_GROUPS
        src = o_grouped[t * O_GROUPS + g : t * O_GROUPS + g + 1, 0:O_GROUP_IN]
        for n0 in pl.range(0, O_LORA, PROJ_A_N_TILE):
            a_acc = pl.create_tensor([H_TILE, PROJ_A_N_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, O_GROUP_IN, PROJ_K_TILE, stage=2):
                a_input = pl.slice(
                    src,
                    [H_TILE, PROJ_K_TILE],
                    [0, k0],
                    valid_shape=[1, PROJ_K_TILE],
                )
                a_weight = wo_a_flat[
                    g * O_LORA + n0 : g * O_LORA + n0 + PROJ_A_N_TILE,
                    k0 : k0 + PROJ_K_TILE,
                ]
                a_acc = pl.matmul_acc(a_acc, a_input, a_weight, b_trans=True, init_cond=(k0 == 0))
            a_result = pl.set_validshape(a_acc, 1, PROJ_A_N_TILE)
            o_lora_flat[t * O_GROUPS + g : t * O_GROUPS + g + 1, n0 : n0 + PROJ_A_N_TILE] = pl.cast(
                a_result, target_type=pl.BF16, mode="rint"
            )

    o_lora = pl.reshape(o_lora_flat, [token_dim, O_GROUPS * O_LORA])
    d_tiles = D // PROJ_D_TILE
    with pl.spmd(token_dim * d_tiles, name_hint="swa_o_proj_b", deps=[pa_tid]) as pb_tid:
        idx = pl.tile.get_block_idx()
        t = idx // d_tiles
        d_tile = idx - t * d_tiles
        d0 = d_tile * PROJ_D_TILE
        b_acc = pl.create_tensor([H_TILE, PROJ_D_TILE], dtype=pl.FP32)
        for k0 in pl.pipeline(0, O_GROUPS * O_LORA, PROJ_K_TILE, stage=2):
            b_input = pl.slice(
                o_lora,
                [H_TILE, PROJ_K_TILE],
                [t, k0],
                valid_shape=[1, PROJ_K_TILE],
            )
            b_weight = wo_b[d0 : d0 + PROJ_D_TILE, k0 : k0 + PROJ_K_TILE]
            b_acc = pl.matmul_acc(b_acc, b_input, b_weight, b_trans=True, init_cond=(k0 == 0))
        b_result = pl.set_validshape(b_acc, 1, PROJ_D_TILE)
        attn_out[t : t + 1, d0 : d0 + PROJ_D_TILE] = pl.cast(b_result, target_type=pl.BF16, mode="rint")
    return attn_out


@pl.jit.inline
def decode_swa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    window_kv_cache: pl.Tensor[[B, WIN, HEAD_DIM], pl.BF16],
    window_topk: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_lens: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
):
    """Write the current decode KV row, then run the SWA attention path."""
    cache = write_decode_window_kv(window_kv_cache, kv, start_pos)
    return sparse_attn_swa(
        q,
        cache,
        window_topk,
        window_lens,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        attn_out,
    )


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    window_kv_cache: pl.InOut[pl.Tensor[[B, WIN, HEAD_DIM], pl.BF16]],
    window_topk: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_lens: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    return sparse_attn_swa(
        q,
        window_kv_cache,
        window_topk,
        window_lens,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        attn_out,
    )


def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    batch: int = B,
    start_pos: int = WIN,
    seed: int = 0,
):
    """Build deterministic tensors for the standalone golden harness.

    ``start_pos`` selects the decode ring order used by the Hugging Face
    implementation.  The cache itself is already populated because
    ``sparse_attn_test`` is the attention-only entry; ``decode_swa`` is the
    entry to use when the current KV row must be written first.
    """
    import torch

    from golden import TensorSpec

    if not 1 <= batch <= B:
        raise ValueError(f"batch must be in [1, {B}], got {batch}")
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")

    generator = torch.Generator().manual_seed(seed)
    metadata = get_window_topk_idxs(WIN, batch, S, start_pos)
    metadata = pad_window_topk(metadata, WIN).reshape(batch * S, WIN)
    if short_window_fixture:
        # Keep the newest 17 ring rows and leave the older physical slots
        # invalid.  This exercises the non-prefix valid-slot mask.
        metadata[:, :-17] = -1
    lens = (metadata >= 0).sum(dim=-1).to(torch.int32)
    tokens = batch * S

    def init_q():
        q = torch.rand((tokens, H, HEAD_DIM), generator=generator, dtype=torch.float32) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q.to(torch.bfloat16)

    def init_cache():
        values = torch.rand((B, WIN, HEAD_DIM), generator=generator, dtype=torch.float32) - 0.5
        return values.to(torch.bfloat16)

    def init_wo_a():
        values = torch.rand(
            (O_GROUPS, O_LORA, O_GROUP_IN),
            generator=generator,
            dtype=torch.float32,
        ) - 0.5
        return (values / (O_GROUP_IN**0.5)).to(torch.bfloat16)

    def init_wo_b():
        values = torch.rand(
            (D, O_GROUPS * O_LORA),
            generator=generator,
            dtype=torch.float32,
        ) - 0.5
        return (values / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)

    def init_cos():
        angles = torch.arange(tokens * HALF_ROPE, dtype=torch.float32).reshape(tokens, HALF_ROPE) * 1e-3
        half = torch.cos(angles)
        return torch.cat([half, half], dim=-1).to(torch.bfloat16)

    def init_sin():
        angles = torch.arange(tokens * HALF_ROPE, dtype=torch.float32).reshape(tokens, HALF_ROPE) * 1e-3
        half = torch.sin(angles)
        return torch.cat([half, half], dim=-1).to(torch.bfloat16)

    return [
        TensorSpec("q", [tokens, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec(
            "window_kv_cache",
            [B, WIN, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cache,
        ),
        TensorSpec(
            "window_topk",
            [tokens, WIN],
            torch.int32,
            init_value=lambda: metadata.clone(),
        ),
        TensorSpec(
            "window_lens",
            [tokens],
            torch.int32,
            init_value=lambda: lens.clone(),
        ),
        TensorSpec("attn_sink", [H], torch.float32, init_value=torch.zeros(H)),
        TensorSpec("freqs_cos", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec(
            "wo_a",
            [O_GROUPS, O_LORA, O_GROUP_IN],
            torch.bfloat16,
            init_value=init_wo_a,
        ),
        TensorSpec(
            "wo_b",
            [D, O_GROUPS * O_LORA],
            torch.bfloat16,
            init_value=init_wo_b,
        ),
        TensorSpec("attn_out", [tokens, D], torch.bfloat16),
    ]


def main(argv=None):
    """Run the standalone PTO golden harness from the command line."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash standalone SWA decode")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=B,
        help=f"runtime request count in [1, {B}]",
    )
    parser.add_argument("--start-pos", type=int, default=WIN, help="decode position used for ring ordering")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--causal-regression-fixture",
        action="store_true",
        help="make the first query numerically dominant for cache-order debugging",
    )
    parser.add_argument(
        "--short-window-fixture",
        action="store_true",
        help="keep only the newest 17 physical ring rows",
    )
    parser.add_argument("--save-data", action="store_true", help="save inputs and golden outputs for replay")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument(
        "--enable-chip-swimlane",
        type=int,
        nargs="?",
        const=1,
        default=0,
        choices=range(5),
    )
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.batch <= B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")
    if args.start_pos < 0:
        parser.error(f"--start-pos must be non-negative, got {args.start_pos}")

    print(f"TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)
    result = run(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            causal_regression_fixture=args.causal_regression_fixture,
            short_window_fixture=args.short_window_fixture,
            batch=args.batch,
            start_pos=args.start_pos,
            seed=args.seed,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        save_data=args.save_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
