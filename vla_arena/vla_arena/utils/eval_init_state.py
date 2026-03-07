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

from __future__ import annotations

from typing import Literal

import numpy as np


InitStateSelectionMode = Literal['first', 'episode_idx']


def select_init_state_index(
    *,
    num_initial_states: int,
    episode_idx: int,
    selection_mode: str,
    offset: int = 0,
    offset_random: bool = False,
    rng: np.random.Generator | None = None,
) -> int | None:
    if num_initial_states <= 0:
        return None

    if selection_mode == 'first':
        base_index = 0
    elif selection_mode == 'episode_idx':
        base_index = episode_idx
    else:
        raise ValueError(
            "Unsupported init_state_selection_mode: "
            f"{selection_mode}. Expected one of: 'first', 'episode_idx'."
        )

    random_offset = 0
    if offset_random:
        if rng is None:
            raise ValueError('rng must be provided when offset_random is True')
        random_offset = int(rng.integers(0, num_initial_states))

    return int((base_index + offset + random_offset) % num_initial_states)
