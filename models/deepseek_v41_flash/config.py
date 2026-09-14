# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The released DeepSeek-V4.1-Flash text-attention configuration.

The attention fields consumed by the standalone SWA kernel are kept here,
together with the published layer metadata needed to identify the SWA layers.
The remaining V4.1 components (CSA2, Engram, and DSpark) have separate runtime
contracts and are intentionally outside this operator directory.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV41FlashConfig:
    hidden_size: int = 5120
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    rms_norm_eps: float = 1e-20
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_max_position_embeddings: int = 65_536
    num_hidden_layers: int = 40
    num_nextn_predict_layers: int = 3
    # Two SWA-only layers, eighteen ratio-2 layers, twenty ratio-1 layers,
    # and three DSpark layers (the 43 entries include MTP layers).
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


FLASH = DeepSeekV41FlashConfig()

# The standalone decode fixture uses one query per request.  The kernel keeps
# the token axis explicit so callers can raise these limits without changing
# the attention math.
DECODE_BATCH = 4
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
BLOCK_SIZE = 128
DECODE_ORI_BLOCK_NUM = DECODE_BATCH
