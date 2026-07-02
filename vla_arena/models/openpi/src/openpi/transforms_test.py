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

import numpy as np
import vla_arena.models.openpi.src.openpi.models.tokenizer as _tokenizer
import vla_arena.models.openpi.src.openpi.shared.normalize as _normalize
import vla_arena.models.openpi.src.openpi.transforms as _transforms
import pytest


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            'a': {'b': 'b/c'},
            'd': 'e/f',
        }
    )
    item = {'b': {'c': 1}, 'e': {'f': 2}}
    assert transform(item) == {'a': {'b': 1}, 'd': 2}


def test_delta_actions():
    item = {
        'state': np.array([1, 2, 3]),
        'actions': np.array([[3, 4, 5], [5, 6, 7]]),
    }

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed['state'] == np.array([1, 2, 3]))
    assert np.all(transformed['actions'] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {
        'state': np.array([1, 2, 3]),
        'actions': np.array([[3, 4, 5], [5, 6, 7]]),
    }

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item['actions']
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {
        'state': np.array([1, 2, 3]),
        'actions': np.array([[3, 4, 5], [5, 6, 7]]),
    }

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed['state'] == np.array([1, 2, 3]))
    assert np.all(transformed['actions'] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {
        'state': np.array([1, 2, 3]),
        'actions': np.array([[3, 4, 5], [5, 6, 7]]),
    }

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item['actions']
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (
        True,
        True,
        False,
        False,
        True,
        True,
    )
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({'prompt': 'Hello, world!'})

    tok_prompt, tok_mask = tokenizer.tokenize('Hello, world!')
    assert np.allclose(tok_prompt, data['tokenized_prompt'])
    assert np.allclose(tok_mask, data['tokenized_prompt_mask'])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match='Prompt is required'):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {'a': {'b': 1, 'c': 2}}
    output = _transforms.transform_dict({'a/b': 'a/c', 'a/c': None}, input)
    assert output == {'a': {'c': 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({'a/b': 'a/c'}, input)

    # Full match is required and so nothing will be removed.
    input = {'a': {'b': 1, 'c': 2}}
    output = _transforms.transform_dict({'a': None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {'a': {'b': 1, 'c': 2}}
    output = _transforms.transform_dict({'a.+': None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {'a': {'b': 1, 'c': 1}, 'b': {'c': 2}}
    output = _transforms.transform_dict({'(.+)/c': r'\1/d'}, input)
    assert output == {'a': {'b': 1, 'd': 1}, 'b': {'d': 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: 'Hello, world!'})

    data = transform({'task_index': 1})
    assert data['prompt'] == 'Hello, world!'

    with pytest.raises(
        ValueError, match='task_index=2 not found in task mapping'
    ):
        transform({'task_index': 2})


def test_quantile_normalize_falls_back_to_minmax_for_sparse_nonzero_dim():
    stats = _normalize.NormStats(
        mean=np.array([0.0, 0.0]),
        std=np.array([1.0, 0.03]),
        q01=np.array([-1.0, 0.0]),
        q99=np.array([1.0, 0.0]),
        min=np.array([-1.0, -0.675]),
        max=np.array([1.0, 0.45]),
    )
    transform = _transforms.Normalize({'actions': stats}, use_quantiles=True)

    data = transform({'actions': np.array([[0.0, 0.45]])})

    assert np.allclose(data['actions'], np.array([[0.0, 1.0]]), atol=1e-5)


def test_quantile_unnormalize_matches_minmax_fallback():
    stats = _normalize.NormStats(
        mean=np.array([0.0, 0.0]),
        std=np.array([1.0, 0.03]),
        q01=np.array([-1.0, 0.0]),
        q99=np.array([1.0, 0.0]),
        min=np.array([-1.0, -0.675]),
        max=np.array([1.0, 0.45]),
    )
    normalize = _transforms.Normalize({'actions': stats}, use_quantiles=True)
    unnormalize = _transforms.Unnormalize({'actions': stats}, use_quantiles=True)
    actions = np.array([[0.0, 0.45], [0.5, -0.675]])

    normalized = normalize({'actions': actions})['actions']
    recovered = unnormalize({'actions': normalized})['actions']

    assert np.allclose(recovered, actions, atol=1e-5)


def test_quantile_normalize_keeps_legacy_behavior_without_minmax_stats():
    stats = _normalize.NormStats(
        mean=np.array([0.0]),
        std=np.array([1.0]),
        q01=np.array([0.0]),
        q99=np.array([0.0]),
    )
    transform = _transforms.Normalize({'actions': stats}, use_quantiles=True)

    data = transform({'actions': np.array([[0.45]])})

    assert data['actions'][0, 0] > 100000


def test_quantile_normalize_constant_dim_with_minmax_stats_outputs_zero():
    stats = _normalize.NormStats(
        mean=np.array([0.0]),
        std=np.array([0.0]),
        q01=np.array([0.0]),
        q99=np.array([0.0]),
        min=np.array([0.0]),
        max=np.array([0.0]),
    )
    normalize = _transforms.Normalize({'actions': stats}, use_quantiles=True)
    unnormalize = _transforms.Unnormalize({'actions': stats}, use_quantiles=True)

    normalized = normalize({'actions': np.array([[0.0]])})['actions']
    recovered = unnormalize({'actions': normalized})['actions']

    assert np.allclose(normalized, np.array([[0.0]]))
    assert np.allclose(recovered, np.array([[0.0]]))
