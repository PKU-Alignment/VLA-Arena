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
from einops import rearrange, repeat
from latent_action_model.genie.modules.blocks import (
    SpatioTemporalTransformer,
    SpatioTransformer,
    VectorQuantizer,
)
from torch import Tensor
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class UncontrolledDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size**2

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        self.dino_encoder = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14_reg'
        )
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(
            torch.empty(1, 1, self.num_codes, dino_dim)
        )  # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,  # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load T5 text encoder model
        self.text_encoder = T5EncoderModel.from_pretrained('./t5-base')
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)

        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('./t5-base')

    def encode_text(self, lang: list):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors='pt', padding=True).to(
            self.device
        )

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask

    def vq_encode(
        self,
        videos: Tensor,
        lang_embed: Tensor = None,
        attention_mask: Tensor = None,
    ) -> dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, 'b T c h w -> (b T) c h w')
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)[
            'x_norm_patchtokens'
        ]
        dion_features = rearrange(dion_features, '(b T) l d -> b T l d', T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask)

        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, : self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            'patches': dion_features,
            'z_q': z_q,
            'z': z,
            'emb': emb,
            'indices': indices,
        }

    def forward(self, batch: dict) -> dict:
        # Encode + VQ
        B, T = batch['videos'].shape[:2]
        H, W = batch['videos'].shape[3:5]

        lang_embed, attention_mask = self.encode_text(
            batch['task_instruction']
        )
        lang_embed = self.lang_proj(lang_embed)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (B, self.num_codes + (H // self.patch_size) ** 2)
                ).to(self.device),
                attention_mask,
            ],
            dim=-1,
        )

        outputs = self.vq_encode(
            batch['videos'],
            repeat(lang_embed, 'b l d -> b T l d', T=T),
            attention_mask.repeat(T, 1),
        )
        video_patches = self.patch_up(outputs['patches'][:, :-1])
        action_patches = self.action_up(outputs['z_q'])
        video_action_patches = torch.cat(
            [action_patches, video_patches], dim=2
        )

        # Decode
        video_recon = self.decoder(
            video_action_patches, lang_embed.unsqueeze(1), attention_mask
        )
        video_recon = video_recon[
            :, :, self.num_codes : self.num_codes + video_patches.shape[2]
        ]

        outputs.update(
            {'recon': video_recon, 'target': outputs['patches'][:, [-1]]}
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size**2

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        self.dino_encoder = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14_reg'
        )
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(
            torch.empty(1, 1, self.num_codes, dino_dim)
        )  # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,  # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.action_latent_controllable = nn.Parameter(
            torch.empty(1, 1, self.num_codes, dino_dim)
        )
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)

    def vq_encode(
        self,
        videos: Tensor,
        lang_embed: Tensor = None,
        attention_mask: Tensor = None,
    ) -> dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, 'b T c h w -> (b T) c h w')
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)[
            'x_norm_patchtokens'
        ]
        dion_features = rearrange(dion_features, '(b T) l d -> b T l d', T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(
            B, T, -1, -1
        )
        padded_patches = torch.cat(
            [action_pad_controllable, padded_patches], dim=2
        )

        # Encode
        z = self.encoder(padded_patches)

        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(
            z[:, 1:, self.num_codes : self.num_codes * 2]
        )

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(
            B * (T - 1), self.num_codes, self.latent_dim
        )
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(
            z_uncontrol
        )
        z_q_uncontrol = z_q_uncontrol.reshape(
            B, T - 1, self.num_codes, self.latent_dim
        )

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(
            z[:, 1:, : self.num_codes]
        )  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(
            B * (T - 1), self.num_codes, self.latent_dim
        )
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            'patches': dion_features,
            'z_q': z_q,
            'z': z,
            'emb': emb,
            'z_q_uncontrol': z_q_uncontrol,
            'z_uncontrol': z_uncontrol,
            'emb_uncontrol': emb_uncontrol,
            'indices': indices,
            'indices_uncontrol': indices_uncontrol,
        }

    def forward(self, batch: dict) -> dict:
        # Encode + VQ
        B, T = batch['videos'].shape[:2]
        H, W = batch['videos'].shape[3:5]

        outputs = self.vq_encode(batch['videos'])
        video_patches = self.patch_up(outputs['patches'][:, :-1])

        # Decode
        video_action_patches = torch.cat(
            [
                self.action_up(outputs['z_q']),
                self.action_up_uncontrol(outputs['z_q_uncontrol']),
                video_patches,
            ],
            dim=2,
        )
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2] :]

        outputs.update(
            {'recon': video_recon, 'target': outputs['patches'][:, [-1]]}
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
