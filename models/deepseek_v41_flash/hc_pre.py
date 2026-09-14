# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-Pass mHC pre-mix for DeepSeek-V4.1-Flash."""

import os
import sys

import pypto.language as pl

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


T_DYN = pl.dynamic("T_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
T_TILE = 8
D_TILE = 256
D_SPMD = 1024

assert HC_MULT == 4
assert D % D_TILE == 0
assert D % D_SPMD == 0


@pl.jit.inline
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the parallel residual copies into one BF16 sublayer input."""
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE * (D // D_SPMD), name_hint="hc_pre"):
        token_block = block // (D // D_SPMD)
        d_block = block % (D // D_SPMD)
        t0 = token_block * T_TILE
        d_base = d_block * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile = pl.load(
            pre,
            [t0, 0],
            [T_TILE, 8],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        pre_transposed = pl.transpose(pre_tile, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_transposed[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_transposed[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_transposed[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_transposed[3:4, 0:T_TILE], [T_TILE, 1])

        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.cast(
                pl.load(x_flat, [t0, d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            )
            x1 = pl.cast(
                pl.load(x_flat, [t0, D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            )
            x2 = pl.cast(
                pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            )
            x3 = pl.cast(
                pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            )
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            mixed_fp32 = pl.add(y0, y1)
            mixed_fp32 = pl.add(mixed_fp32, y2)
            mixed_fp32 = pl.add(mixed_fp32, y3)
            mixed = pl.cast(
                mixed_fp32,
                target_type=pl.BF16,
                mode="rint",
            )
            mixed_valid = pl.set_validshape(mixed, valid_rows, D_TILE)
            pl.store(mixed_valid, [t0, d0], x_mixed)
    return x_mixed


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    pre.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    hc_pre(x, pre, x_mixed)
    return x_mixed


def golden_hc_pre(tensors):
    """Torch reference for the official BF16 pre-mix boundary."""
    import torch

    x = tensors["x"].float()
    pre = tensors["pre"].float()
    mixed = torch.sum(pre.unsqueeze(-1) * x, dim=1)
    tensors["x_mixed"][:] = mixed.to(torch.bfloat16)


def exact_hc_pre_compare(actual, expected, **_kwargs):
    """Require bitwise equality with the PyTorch BF16 reference."""
    import torch

    if torch.equal(actual, expected):
        return True, ""
    mismatch = torch.where(actual.flatten() != expected.flatten())[0]
    first = mismatch[:1]
    index = int(first[0].item()) if first.numel() else -1
    return False, (
        f"    bitwise mismatch count={mismatch.numel()}/{actual.numel()}"
        f" first_index={index} actual={actual.flatten()[index].item()}"
        f" expected={expected.flatten()[index].item()}"
    )


def build_tensor_specs(batch=DECODE_BATCH, sequence=DECODE_SEQ):
    import torch
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(1)

    def init_x():
        return (torch.randn(tokens, HC_MULT, D, generator=generator) * 0.05).to(torch.bfloat16)

    def init_pre():
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator)) + 1e-6

    return [
        TensorSpec("x", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("pre", [tokens, HC_MULT], torch.float32, init_value=init_pre),
        TensorSpec("x_mixed", [tokens, D], torch.bfloat16),
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
        fn=hc_pre_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_pre,
        config={"platform": args.platform, "device_id": args.device},
        compare_fn={"x_mixed": exact_hc_pre_compare},
        rtol=0.0,
        atol=0.0,
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
