# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-Pass mHC post-mix for DeepSeek-V4.1-Flash."""

import pypto.language as pl
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


T_DYN = pl.dynamic("T_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
D_TILE = 256

assert HC_MULT == 4
assert D % D_TILE == 0


@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Expand a sublayer result and mix the previous residual streams."""
    t_dim = pl.tensor.dim(x, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    for block in pl.spmd(t_dim * HC_MULT, name_hint="hc_post"):
        t = block // HC_MULT
        out_h = block % HC_MULT
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x_tile = pl.cast(x[t : t + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
            comb_w = pl.read(comb, [t, out_h])
            residual_tile = pl.cast(
                residual_flat[t : t + 1, d0 : d0 + D_TILE],
                target_type=pl.FP32,
            )
            residual_value = pl.mul(residual_tile, comb_w)
            for in_h in pl.unroll(HC_MULT - 1):
                in_idx = in_h + 1
                comb_w = pl.read(comb, [t, in_idx * HC_MULT + out_h])
                residual_tile = pl.cast(
                    residual_flat[t : t + 1, in_idx * D + d0 : in_idx * D + d0 + D_TILE],
                    target_type=pl.FP32,
                )
                residual_value = pl.add(residual_value, pl.mul(residual_tile, comb_w))
            post_w = pl.read(post, [t, out_h])
            value = pl.add(pl.mul(x_tile, post_w), residual_value)
            y_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = pl.cast(
                value, target_type=pl.BF16, mode="rint"
            )
    return y


@pl.jit.inline
def hc_post_prefill(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run post-mix on the active prefix and clear the static tail."""
    t_dim = pl.tensor.dim(x, 0)
    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    if active > t_dim:
        active = t_dim
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    for block in pl.spmd(active * HC_MULT, name_hint="hc_post_prefill"):
        t = block // HC_MULT
        out_h = block % HC_MULT
        if t < active:
            for d0 in pl.pipeline(0, D, D_TILE, stage=2):
                x_tile = pl.cast(x[t : t + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                comb_w = pl.read(comb, [t, out_h])
                residual_tile = pl.cast(
                    residual_flat[t : t + 1, d0 : d0 + D_TILE],
                    target_type=pl.FP32,
                )
                residual_value = pl.mul(residual_tile, comb_w)
                for in_h in pl.unroll(HC_MULT - 1):
                    in_idx = in_h + 1
                    comb_w = pl.read(comb, [t, in_idx * HC_MULT + out_h])
                    residual_tile = pl.cast(
                        residual_flat[t : t + 1, in_idx * D + d0 : in_idx * D + d0 + D_TILE],
                        target_type=pl.FP32,
                    )
                    residual_value = pl.add(residual_value, pl.mul(residual_tile, comb_w))
                post_w = pl.read(post, [t, out_h])
                value = pl.add(pl.mul(x_tile, post_w), residual_value)
                y_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = pl.cast(
                    value, target_type=pl.BF16, mode="rint"
                )

    inactive = t_dim - active
    for block in pl.spmd(((inactive + 15) // 16) * HC_MULT, name_hint="hc_post_prefill_zero"):
        tile = block // HC_MULT
        out_h = block % HC_MULT
        t0 = active + tile * 16
        zero = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
        for dt in pl.range(16):
            t = t0 + dt
            if t < t_dim:
                for d0 in pl.range(0, D, D_TILE):
                    y_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = zero
    return y


@pl.jit
def hc_post_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    hc_post(x, residual, post, comb, y)
    return y


def golden_hc_post(tensors):
    """Torch reference for the official BF16 post-mix boundary."""
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post = tensors["post"].float()
    comb = tensors["comb"].float().reshape(-1, HC_MULT, HC_MULT)
    residual_value = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    value = post.unsqueeze(-1) * x.unsqueeze(-2) + residual_value
    tensors["y"][:] = value.to(torch.bfloat16)


def exact_hc_post_compare(actual, expected, **_kwargs):
    """Require bitwise equality with the PyTorch BF16 reference."""
    import torch

    if torch.equal(actual, expected):
        return True, ""
    mismatch = torch.where(actual.flatten() != expected.flatten())[0]
    first = mismatch[:1]
    index = int(first[0]) if first.numel() else -1
    return False, (
        f"    bitwise mismatch count={mismatch.numel()}/{actual.numel()}"
        f" first_index={index} actual={actual.flatten()[index].item()}"
        f" expected={expected.flatten()[index].item()}"
    )


def build_tensor_specs(batch=DECODE_BATCH, sequence=DECODE_SEQ):
    import torch
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(2)

    def init_x():
        return (torch.randn(tokens, D, generator=generator) * 0.05).to(torch.bfloat16)

    def init_residual():
        return (torch.randn(tokens, HC_MULT, D, generator=generator) * 0.05).to(torch.bfloat16)

    def init_post():
        return 2.0 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))

    def init_comb():
        value = torch.rand(tokens, HC_MULT, HC_MULT, generator=generator) + 0.1
        return (value / value.sum(dim=-2, keepdim=True)).reshape(tokens, HC_MULT * HC_MULT)

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_residual),
        TensorSpec("post", [tokens, HC_MULT], torch.float32, init_value=init_post),
        TensorSpec("comb", [tokens, HC_MULT * HC_MULT], torch.float32, init_value=init_comb),
        TensorSpec("y", [tokens, HC_MULT, D], torch.bfloat16),
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
        fn=hc_post_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_post,
        config={"platform": args.platform, "device_id": args.device},
        compare_fn={"y": exact_hc_post_compare},
        rtol=1e-3,
        atol=1e-4,
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
