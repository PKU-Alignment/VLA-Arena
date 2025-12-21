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

import flax.nnx as nnx
import jax
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(
        abstract_model, nnx.All(nnx.Param, freeze_filter)
    ).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant='gemma_2b_lora')
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all('lora' not in p for p in state)
    assert all('llm' in p for p in state)
    assert all('_1' not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant='gemma_300m_lora')
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all('lora' not in p for p in state)
    assert all('llm' in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any('_1' in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(
        paligemma_variant='gemma_2b_lora',
        action_expert_variant='gemma_300m_lora',
    )
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all('lora' not in p for p in state)
    assert all('llm' in p for p in state)
