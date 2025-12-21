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

import pytest
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import action_chunk_broker


@pytest.mark.manual
def test_infer():
    config = _config.get_config('pi0_aloha_sim')
    policy = _policy_config.create_trained_policy(
        config, 'gs://openpi-assets/checkpoints/pi0_aloha_sim'
    )

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result['actions'].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config('pi0_aloha_sim')
    policy = _policy_config.create_trained_policy(
        config, 'gs://openpi-assets/checkpoints/pi0_aloha_sim'
    )

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs['actions'].shape == (14,)
