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

"""
in1k_vit.py

Vision Transformers trained / finetuned on ImageNet (ImageNet-21K =>> ImageNet-1K)
"""

from vla_arena.models.openvla_oft.prismatic.models.backbones.vision.base_vision import (
    TimmViTBackbone,
)


# Registry =>> Supported Vision Backbones (from TIMM)
IN1K_VISION_BACKBONES = {
    'in1k-vit-l': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
}


class IN1KViTBackbone(TimmViTBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
    ) -> None:
        super().__init__(
            vision_backbone_id,
            IN1K_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
