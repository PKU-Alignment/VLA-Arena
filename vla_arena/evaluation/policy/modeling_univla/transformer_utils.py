import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.init import constant_, xavier_uniform_
from einops import rearrange, repeat
import numpy as np
# from torch import einsum


# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
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
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor, attention_mask = None) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"

        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        # print(scores.shape)
        if attention_mask is not None:
            attention_mask = (
                attention_mask[None, None, :, :].repeat(1, self.n_heads, 1, 1) #.flatten(0, 1)
            )
            scores.masked_fill_(attention_mask == 0, float("-inf"))
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
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, n_heads

        # Projection Operator
        self.projection = nn.Linear(vis_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: torch.Tensor, mask = None, init_embed = None) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = latents + init_embed.unsqueeze(1) if init_embed is not None else latents
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x), mask))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)
        
class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size = 5):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)

        self.proj = nn.Sequential(
                                nn.Linear(512, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens, init_embed = visual_embed))
        
        return action


class ActionDecoder(nn.Module):
    def __init__(self,window_size=5):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)

        self.temporal_size = window_size
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]


    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    
    def forward(self, latent_actions, visual_embed, mask, action_low, action_high):
        # Forward action decoder
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float)).reshape(-1, self.temporal_size, 7)
        pred_action = np.array(pred_action.tolist())
        
        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict actions
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
        
        action_prediction = np.where(
            mask,
            0.5 * (action_prediction + 1) * (action_high - action_low) + action_low,
            action_prediction,
        )

        return action_prediction
