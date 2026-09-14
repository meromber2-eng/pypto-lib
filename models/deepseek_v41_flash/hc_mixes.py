# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-Pass mHC coefficient generation for DeepSeek-V4.1-Flash.

The reference implementation projects the flattened four-stream residual into
24 values, then splits those values into ``pre``, ``post`` and ``comb``.  The
projection and all nonlinear operations stay in FP32; only the residual stream
itself is BF16 at the surrounding block boundary.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


T_DYN = pl.dynamic("T_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 8
RMS_K_TILE = 512
LINEAR_K_TILE = 256
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

assert HC_MULT == 4
assert HC_DIM % RMS_K_TILE == 0
assert HC_DIM % LINEAR_OK == 0
assert LINEAR_K_PER_SPLIT % LINEAR_K_TILE == 0


@pl.jit.inline
def hc_mixes(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Project one flattened residual stream and produce all mHC coefficients."""
    t_dim = pl.tensor.dim(x, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_mixes_rms"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            rms_x_tile = pl.slice(
                x_flat,
                [T_TILE, RMS_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, RMS_K_TILE],
            )
            rms_x_fp32 = pl.cast(rms_x_tile, target_type=pl.FP32)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_fp32, rms_x_fp32)), [1, T_TILE]))
        rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
            pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
        )

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_mixes_linear"):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        split = task % LINEAR_OK
        k_base = split * LINEAR_K_PER_SPLIT
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            linear_x_tile = pl.slice(
                x_flat,
                [LINEAR_T_TILE, LINEAR_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, LINEAR_K_TILE],
            )
            linear_x_fp32 = pl.cast(linear_x_tile, target_type=pl.FP32)
            w_tile = pl.slice(
                hc_fn,
                [MIX_PAD, LINEAR_K_TILE],
                [0, k0],
                valid_shape=[MIX_HC, LINEAR_K_TILE],
            )
            acc = pl.matmul_acc(acc, linear_x_fp32, w_tile, b_trans=True, init_cond=(kb == 0))
        partial = split * t_linear + t0
        mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD] = acc

    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_mixes_linear_reduce"):
        t0 = block * LINEAR_T_TILE
        total = mixes_partials[t0 : t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for split in pl.range(1, LINEAR_OK):
            partial = split * t_linear + t0
            total = pl.add(total, mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD])
        mixes_raw[t0 : t0 + LINEAR_T_TILE, 0:MIX_PAD] = total

    pre_tail = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    post_tail = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    base_view = pl.reshape(hc_base, [1, MIX_HC])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_mixes_split"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        inv = inv_rms[t0 : t0 + T_TILE, 0:1]
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_logits = pl.add(
            pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv), scale0),
            pl.col_expand(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], pre_base),
        )
        pre_value = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
        post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
        post_logits = pl.add(
            pl.mul(
                pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv),
                scale1,
            ),
            pl.col_expand(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], post_base),
        )
        post_value = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
        if valid_rows == T_TILE:
            pre[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
                pre_value, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT]
            )
            post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
                post_value, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT]
            )
        else:
            pre_tail[0:T_TILE, 0:HC_PAD] = pre_value
            post_tail[0:T_TILE, 0:HC_PAD] = post_value
            pre_out = pl.load(pre_tail, [0, 0], [T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            post_out = pl.load(post_tail, [0, 0], [T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            pl.store(pre_out, [t0, 0], pre)
            pl.store(post_out, [t0, 0], post)

    comb_tail = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    for block in pl.spmd((t_dim + COMB_T_TILE - 1) // COMB_T_TILE, name_hint="hc_mixes_sinkhorn"):
        t0 = block * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        comb_inv = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec)
        comb_offset = HC_MULT * 2
        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        mix0 = pl.load(mixes_raw, [t0, comb_offset + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix1 = pl.load(mixes_raw, [t0, comb_offset + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix2 = pl.load(mixes_raw, [t0, comb_offset + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix3 = pl.load(mixes_raw, [t0, comb_offset + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        base0 = pl.load(base_view, [0, comb_offset + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        base1 = pl.load(base_view, [0, comb_offset + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        base2 = pl.load(base_view, [0, comb_offset + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        base3 = pl.load(base_view, [0, comb_offset + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        logits0 = pl.fillpad(pl.add(pl.mul(pl.row_expand_mul(mix0, comb_inv), scale2), pl.col_expand(mix0, base0)), pad_value=pl.PadValue.min)
        logits1 = pl.fillpad(pl.add(pl.mul(pl.row_expand_mul(mix1, comb_inv), scale2), pl.col_expand(mix1, base1)), pad_value=pl.PadValue.min)
        logits2 = pl.fillpad(pl.add(pl.mul(pl.row_expand_mul(mix2, comb_inv), scale2), pl.col_expand(mix2, base2)), pad_value=pl.PadValue.min)
        logits3 = pl.fillpad(pl.add(pl.mul(pl.row_expand_mul(mix3, comb_inv), scale2), pl.col_expand(mix3, base3)), pad_value=pl.PadValue.min)
        exp0 = pl.exp(pl.row_expand_sub(logits0, pl.row_max(logits0, row_max_tmp)))
        exp1 = pl.exp(pl.row_expand_sub(logits1, pl.row_max(logits1, row_max_tmp)))
        exp2 = pl.exp(pl.row_expand_sub(logits2, pl.row_max(logits2, row_max_tmp)))
        exp3 = pl.exp(pl.row_expand_sub(logits3, pl.row_max(logits3, row_max_tmp)))
        row0 = pl.add(pl.row_expand_div(exp0, pl.row_sum(exp0, row_sum_tmp)), HC_EPS)
        row1 = pl.add(pl.row_expand_div(exp1, pl.row_sum(exp1, row_sum_tmp)), HC_EPS)
        row2 = pl.add(pl.row_expand_div(exp2, pl.row_sum(exp2, row_sum_tmp)), HC_EPS)
        row3 = pl.add(pl.row_expand_div(exp3, pl.row_sum(exp3, row_sum_tmp)), HC_EPS)
        row0 = pl.fillpad(pl.set_validshape(row0, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        row1 = pl.fillpad(pl.set_validshape(row1, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        row2 = pl.fillpad(pl.set_validshape(row2, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        row3 = pl.fillpad(pl.set_validshape(row3, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        col_sum = pl.add(pl.add(row0, row1), pl.add(row2, row3))
        col_sum = pl.add(col_sum, HC_EPS)
        row0 = pl.div(row0, col_sum)
        row1 = pl.div(row1, col_sum)
        row2 = pl.div(row2, col_sum)
        row3 = pl.div(row3, col_sum)
        sinkhorn_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        for _ in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_sum = pl.add(pl.row_sum(row0, sinkhorn_sum_tmp), HC_EPS)
            row1_sum = pl.add(pl.row_sum(row1, sinkhorn_sum_tmp), HC_EPS)
            row2_sum = pl.add(pl.row_sum(row2, sinkhorn_sum_tmp), HC_EPS)
            row3_sum = pl.add(pl.row_sum(row3, sinkhorn_sum_tmp), HC_EPS)
            row0 = pl.row_expand_div(row0, row0_sum)
            row1 = pl.row_expand_div(row1, row1_sum)
            row2 = pl.row_expand_div(row2, row2_sum)
            row3 = pl.row_expand_div(row3, row3_sum)
            col_sum = pl.add(pl.add(row0, row1), pl.add(row2, row3))
            col_sum = pl.add(col_sum, HC_EPS)
            row0 = pl.div(row0, col_sum)
            row1 = pl.div(row1, col_sum)
            row2 = pl.div(row2, col_sum)
            row3 = pl.div(row3, col_sum)

        if valid_rows == COMB_T_TILE:
            pl.store(pl.set_validshape(row0, COMB_T_TILE, HC_MULT), [t0, 0 * HC_MULT], comb)
            pl.store(pl.set_validshape(row1, COMB_T_TILE, HC_MULT), [t0, 1 * HC_MULT], comb)
            pl.store(pl.set_validshape(row2, COMB_T_TILE, HC_MULT), [t0, 2 * HC_MULT], comb)
            pl.store(pl.set_validshape(row3, COMB_T_TILE, HC_MULT), [t0, 3 * HC_MULT], comb)
        else:
            pl.store(row0, [0, 0 * HC_PAD], comb_tail)
            pl.store(row1, [0, 1 * HC_PAD], comb_tail)
            pl.store(row2, [0, 2 * HC_PAD], comb_tail)
            pl.store(row3, [0, 3 * HC_PAD], comb_tail)
            row0_tail = pl.load(comb_tail, [0, 0 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            row1_tail = pl.load(comb_tail, [0, 1 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            row2_tail = pl.load(comb_tail, [0, 2 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            row3_tail = pl.load(comb_tail, [0, 3 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)
    return comb


@pl.jit
def hc_mixes_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    pre.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    hc_mixes(x, hc_fn, hc_scale, hc_base, pre, post, comb)
    return comb


_A5_FP32_VECTOR_LANES = 64


def _a5_trowsum(values):
    import torch

    groups = values.reshape(*values.shape[:-1], -1, _A5_FP32_VECTOR_LANES)
    while groups.shape[-1] > 1:
        groups = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = groups[..., 0] + groups[..., 1]
    grouped = groups[..., 0]
    total = torch.zeros_like(grouped[..., :1])
    for i in range(grouped.shape[-1]):
        total = total + grouped[..., i : i + 1]
    return total


def _golden_linear(x_flat, hc_fn):
    import torch

    split_count = HC_DIM // LINEAR_K_PER_SPLIT
    chunks = LINEAR_K_PER_SPLIT // LINEAR_K_TILE
    x_k = x_flat.reshape(x_flat.shape[0], 1, split_count, chunks, LINEAR_K_TILE)
    w_k = hc_fn.reshape(1, MIX_HC, split_count, chunks, LINEAR_K_TILE)
    per_chunk = (x_k * w_k).sum(dim=-1)
    split_partials = []
    for split in range(split_count):
        value = per_chunk[:, :, split, 0]
        for chunk in range(1, chunks):
            value = value + per_chunk[:, :, split, chunk]
        split_partials.append(value)
    mixes = split_partials[0]
    for value in split_partials[1:]:
        mixes = mixes + value
    return mixes


def golden_hc_mixes(tensors):
    """Torch reference for the official mHC projection and Sinkhorn steps."""
    import torch

    x = tensors["x"].float().reshape(-1, HC_DIM)
    hc_fn = tensors["hc_fn"].float()
    scale = tensors["hc_scale"].float()
    base = tensors["hc_base"].float()
    sq_sum = torch.zeros(x.shape[0], 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        sq_sum = sq_sum + _a5_trowsum(x[:, k0 : k0 + RMS_K_TILE] * x[:, k0 : k0 + RMS_K_TILE])
    inv_rms = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)
    mixes = _golden_linear(x, hc_fn) * inv_rms
    pre = torch.sigmoid(mixes[:, :HC_MULT] * scale[0] + base[:HC_MULT]) + HC_EPS
    post = 2.0 * torch.sigmoid(mixes[:, HC_MULT : 2 * HC_MULT] * scale[1] + base[HC_MULT : 2 * HC_MULT])
    comb_value = mixes[:, 2 * HC_MULT :] * scale[2] + base[2 * HC_MULT :]
    comb_value = comb_value.reshape(-1, HC_MULT, HC_MULT)
    comb_value = torch.softmax(comb_value, dim=-1) + HC_EPS
    comb_value = comb_value / (comb_value.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_value = comb_value / (comb_value.sum(dim=-1, keepdim=True) + HC_EPS)
        comb_value = comb_value / (comb_value.sum(dim=-2, keepdim=True) + HC_EPS)
    tensors["pre"][:] = pre
    tensors["post"][:] = post
    tensors["comb"][:] = comb_value.reshape(-1, HC_MULT * HC_MULT)


def build_tensor_specs(batch=DECODE_BATCH, sequence=DECODE_SEQ):
    import torch
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(0)

    def init_x():
        return (torch.randn(tokens, HC_MULT, D, generator=generator) * 0.05).to(torch.bfloat16)

    def init_fn():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) * 0.05

    return [
        TensorSpec("x", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=torch.tensor([0.0761, 0.0326, 0.2270])),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=torch.zeros(MIX_HC)),
        TensorSpec("pre", [tokens, HC_MULT], torch.float32),
        TensorSpec("post", [tokens, HC_MULT], torch.float32),
        TensorSpec("comb", [tokens, HC_MULT * HC_MULT], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run(
        fn=hc_mixes_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_mixes,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=2.5e-5,
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
