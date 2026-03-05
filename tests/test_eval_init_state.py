import os

import numpy as np
import pytest

os.environ.setdefault('LIBERO_CONFIG_PATH', '/tmp/libero')

from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index


def test_select_init_state_index_first_mode():
    idx = select_init_state_index(
        num_initial_states=5,
        episode_idx=3,
        selection_mode='first',
        offset=0,
        offset_random=False,
    )
    assert idx == 0


def test_select_init_state_index_episode_idx_mode():
    idx = select_init_state_index(
        num_initial_states=5,
        episode_idx=7,
        selection_mode='episode_idx',
        offset=0,
        offset_random=False,
    )
    assert idx == 2


def test_select_init_state_index_with_offset():
    idx = select_init_state_index(
        num_initial_states=5,
        episode_idx=0,
        selection_mode='first',
        offset=3,
        offset_random=False,
    )
    assert idx == 3


def test_select_init_state_index_random_offset_range_and_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    values1 = [
        select_init_state_index(
            num_initial_states=7,
            episode_idx=i,
            selection_mode='episode_idx',
            offset=1,
            offset_random=True,
            rng=rng1,
        )
        for i in range(20)
    ]
    values2 = [
        select_init_state_index(
            num_initial_states=7,
            episode_idx=i,
            selection_mode='episode_idx',
            offset=1,
            offset_random=True,
            rng=rng2,
        )
        for i in range(20)
    ]

    assert values1 == values2
    assert all(0 <= value < 7 for value in values1)


def test_select_init_state_index_no_initial_states():
    idx = select_init_state_index(
        num_initial_states=0,
        episode_idx=0,
        selection_mode='first',
        offset=0,
        offset_random=False,
    )
    assert idx is None


def test_select_init_state_index_invalid_mode():
    with pytest.raises(ValueError, match='Unsupported init_state_selection_mode'):
        select_init_state_index(
            num_initial_states=3,
            episode_idx=0,
            selection_mode='invalid_mode',
            offset=0,
            offset_random=False,
        )


def test_select_init_state_index_random_mode_requires_rng():
    with pytest.raises(ValueError, match='rng must be provided'):
        select_init_state_index(
            num_initial_states=3,
            episode_idx=0,
            selection_mode='first',
            offset=0,
            offset_random=True,
            rng=None,
        )
