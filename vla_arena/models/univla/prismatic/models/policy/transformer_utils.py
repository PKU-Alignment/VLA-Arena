# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from einops import repeat


# from torch import einsum


# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            f'invalid input for _is_power_of_2: {n} (type: {type(n)})'
        )
    return (n & (n - 1) == 0) and n != 0


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


# As defined in Set Transformers () -- basically the above, additionally taking in
# a set of $k$ learned "seed vectors" that are used to "pool" information.
class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert (
            embed_dim % n_heads == 0
        ), '`embed_dim` must be divisible by `n_heads`!'
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(
            embed_dim, embed_dim, bias=False
        ), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, seed: torch.Tensor, x: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert (
            C_s == C_x
        ), 'Seed vectors and pool inputs must have the same embedding dimensionality!'

        # Project Seed Vectors to `queries`
        q = (
            self.q(seed)
            .reshape(B_s, K, self.n_heads, C_s // self.n_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x)
            .reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        # print(scores.shape)
        if attention_mask is not None:
            attention_mask = attention_mask[None, None, :, :].repeat(
                1, self.n_heads, 1, 1
            )  # .flatten(0, 1)
            scores.masked_fill_(attention_mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)

        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        vis_dim: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = (
            n_latents,
            embed_dim,
            n_heads,
        )

        # Projection Operator
        self.projection = nn.Linear(vis_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(
            torch.zeros(self.n_latents, self.embed_dim), requires_grad=True
        )
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = (
            RMSNorm(self.embed_dim)
            if do_rms_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-6)
        )
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

        # Position-wise Feed-Forward Components
        self.mlp_norm = (
            RMSNorm(self.embed_dim)
            if do_rms_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-6)
        )
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(
                    nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)),
                    nn.GELU(),
                )
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(
        self, x: torch.Tensor, mask=None, init_embed=None
    ) -> torch.Tensor:
        latents = repeat(
            self.latents, 'n_latents d -> bsz n_latents d', bsz=x.shape[0]
        )
        latents = (
            latents + init_embed.unsqueeze(1)
            if init_embed is not None
            else latents
        )
        latents = self.attn_norm(
            latents + self.attn(latents, self.projection(x), mask)
        )
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)
