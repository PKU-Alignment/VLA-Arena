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

from .base_vision import ImageTransform, VisionBackbone
from .clip_vit import CLIPViTBackbone
from .dinoclip_vit import DinoCLIPViTBackbone
from .dinosiglip_vit import DinoSigLIPViTBackbone
from .dinov2_vit import DinoV2ViTBackbone
from .in1k_vit import IN1KViTBackbone
from .siglip_vit import SigLIPViTBackbone
