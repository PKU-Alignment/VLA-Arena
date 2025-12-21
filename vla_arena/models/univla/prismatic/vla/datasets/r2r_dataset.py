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

import copy
import logging
import os
import pickle
import random
import re
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset


# Constants
Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


def process_rgb(
    episode: dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space['rgb_obs']

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = (
                torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size])
                .byte()
                .permute(0, 3, 1, 2)
            )
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {'rgb_obs': seq_rgb_obs_dict}


def process_depth(
    episode: dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space['depth_obs']
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):

        depth_ob = exp_dim(episode[depth_obs_key].squeeze())
        # print(depth_ob.shape)
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(
                depth_ob[seq_idx : seq_idx + window_size]
            ).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {'depth_obs': seq_depth_obs_dict}


def process_actions(
    episode: dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> dict[str, torch.Tensor]:
    # shape: (N_actions)
    # if len(action_keys) != 1:
    #     raise NotImplementedError
    action_key = observation_space
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if 'actions' in transforms:
            action = transforms['actions']((action, episode['robot_obs']))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(
            episode[action_keys[0]][seq_idx : seq_idx + window_size]
        ).float()
        rel_seq_acts = torch.from_numpy(
            episode[action_keys[1]][seq_idx : seq_idx + window_size]
        ).float()

    return {'actions': seq_acts}


def process_language(
    episode: dict[str, np.ndarray], transforms: dict, with_lang: bool
) -> dict[str, torch.Tensor]:
    seq_lang = {'lang': torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode['language']).float()
        if 'language' in transforms:
            lang = transforms['language'](lang)
        seq_lang['lang'] = lang
    return seq_lang


def get_state_info_dict(
    episode: dict[str, np.ndarray],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        'state_info': {
            'robot_obs': torch.from_numpy(episode['robot_obs']),
            'scene_obs': torch.from_numpy(episode['scene_obs']),
        }
    }


def load_dataset_statistics(train_dataset_dir, val_dataset_dir, transforms):
    """
    Tries to load statistics.yaml in every dataset folder in order to update the transforms hardcoded in the
    hydra config file. If no statistics.yaml exists, nothing is changed

    Args:
        train_dataset_dir: path of the training folder
        val_dataset_dir: path of the validation folder
        transforms: transforms loaded from hydra conf

    Returns:
        transforms: potentially updated transforms
    """
    paths = {'train': train_dataset_dir, 'val': val_dataset_dir}
    for dataset_type in ['train', 'val']:
        try:
            statistics = OmegaConf.load(
                Path(paths[dataset_type]) / 'statistics.yaml'
            )
            # Hack for maintaining two repositories with transforms
            statistics = OmegaConf.create(
                OmegaConf.to_yaml(statistics).replace('calvin_models.', '')
            )
            # this ugly piece of code only exists because OmegaConf actually can't merge ListConfigs.
            # we do not want to override everything, but just the transforms that are specified in both
            # see https://stackoverflow.com/questions/61315623/omegaconf-can-i-influence-how-lists-are-merged
            for modality in transforms[dataset_type]:
                if modality in statistics:
                    conf_transforms = transforms[dataset_type][modality]
                    dataset_transforms = statistics[modality]
                    for dataset_trans in dataset_transforms:
                        exists = False
                        for i, conf_trans in enumerate(conf_transforms):
                            if (
                                dataset_trans['_target_']
                                == conf_trans['_target_']
                            ):
                                exists = True
                                transforms[dataset_type][modality][
                                    i
                                ] = dataset_trans
                                break
                        if not exists:
                            transforms[dataset_type][modality] = ListConfig(
                                [*conf_transforms, dataset_trans]
                            )
        except FileNotFoundError:
            logger.warning('Could not load statistics.yaml')
    return transforms


def lookup_naming_pattern(
    dataset_dir: Path, save_format: str
) -> tuple[tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r'\d+', filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r'\d+', filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


obs_config = DictConfig(
    {
        'rgb_obs': ['rgb_static'],
        'depth_obs': ['depth_static'],
        'state_obs': [],
        'actions': ['actions'],  # rel_actions
        'language': ['language'],
    }
)

prop_state = DictConfig(
    {
        'n_state_obs': 15,
        'keep_indices': [[0, 15]],
        'robot_orientation_idx': [3, 6],
        'normalize': True,
        'normalize_robot_orientation': True,
    }
)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(
            x, grid, padding_mode='zeros', align_corners=False
        )

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n * t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n * t, *base_grid.shape[2:])
        shift = torch.randint(
            1,
            2 * self.pad + 1,
            size=(n * t, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x


class BaseR2RDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = 'lang_annotations',
        num_workers: int = 0,
        key: str = 'lang',
        obs_space: DictConfig = obs_config,
        transforms: dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        sampling_step=1,
        image_size=256,
        with_depth=False,
        action_tokenizer=None,
        base_tokenizer=None,
        image_transform=None,
        prompt_builder_fn=None,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        print('*' * 50)
        print(self.transforms)
        self.with_lang = key == 'lang'
        self.relative_actions = (
            'rel_actions' in self.observation_space['actions']
        )

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size

        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        self.resize_img = torchvision.transforms.Resize(224)
        self.image_transform_lam = torchvision.transforms.ToTensor()

        self.sampling_step = sampling_step
        self.act_step = act_step
        # print('ws {}, min_ws {}, max_ws {}'.format(self.window_size, self.max_window_size, self.min_window_size))
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons

        self.color_aug = torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        assert (
            'validation' in self.abs_datasets_dir.as_posix()
            or 'training' in self.abs_datasets_dir.as_posix()
        )
        self.validation = 'validation' in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        print(f'loading dataset at {self.abs_datasets_dir}')
        logger.info('finished loading dataset')

    def process_rgb(
        self,
        episode: dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space['rgb_obs']
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()

            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {'rgb_obs': seq_rgb_obs_dict}

    def process_language(
        self, episode: dict[str, np.ndarray], transforms: dict, with_lang: bool
    ):
        return {'lang': episode['language']}

    def get_openvla_prompt(
        self, instruction: str, tokenized_action: str = None
    ) -> str:
        # print(tokenized_action)
        return f'In: What action should the robot take to {instruction.lower()}?\nOut:'  # + tokenized_action + "</s>"

    # def __iter__(self,):

    #     for idx in range(len(self.episode_lookup)):
    #         yield self.process_data(idx)
    def __getitem__(
        self, idx: int | tuple[int, int], fixed_seed=False
    ) -> dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                if self.padding_sequence:
                    window_size = self.max_window_size
                else:
                    window_size = self._get_window_size(idx)
                # window_size = self.max_window_size
            else:
                logger.error(
                    f'min_window_size {self.min_window_size} > max_window_size {self.max_window_size}'
                )
                raise ValueError
        else:
            idx, window_size = idx

        # print(window_size)
        extra_frame_num = window_size - self.min_window_size

        sequence = self._get_sequences(idx, window_size, head=False)

        image = copy.deepcopy(sequence['rgb_obs']['rgb_static'].numpy())

        image_vla = Image.fromarray(image[extra_frame_num].astype(np.uint8))
        goal_image = Image.fromarray(image[-1].astype(np.uint8))
        pixel_values = self.image_transform(image_vla)

        initial_pixel_values_hist_list, target_pixel_values_hist_list = (
            None,
            None,
        )
        if extra_frame_num > 0:
            assert (self.max_window_size - self.min_window_size) % (
                self.min_window_size - 1
            ) == 0
            initial_pixel_values_hist_list, target_pixel_values_hist_list = (
                [],
                [],
            )
            for i in range(0, extra_frame_num, self.min_window_size - 1):
                hist_frame_prev = Image.fromarray(image[i].astype(np.uint8))
                hist_frame_goal = Image.fromarray(
                    image[i + self.min_window_size - 1].astype(np.uint8)
                )
                initial_pixel_values_hist = self.image_transform_lam(
                    self.resize_img(hist_frame_prev)
                )
                target_pixel_values_hist = self.image_transform_lam(
                    self.resize_img(hist_frame_goal)
                )
                initial_pixel_values_hist_list.append(
                    initial_pixel_values_hist
                )
                target_pixel_values_hist_list.append(target_pixel_values_hist)

        initial_pixel_values = self.image_transform_lam(
            self.resize_img(image_vla)
        )
        target_pixel_values = self.image_transform_lam(
            self.resize_img(goal_image)
        )

        # # tgt_action = normalized_action[pred_actions:]
        if extra_frame_num > 0:
            action = sequence['actions'][
                extra_frame_num : extra_frame_num + self.min_window_size
            ]
        else:
            action = sequence['actions'][:window_size]

        instruction = sequence['lang']

        dataset_name = 'R2R'

        return dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist_list,
            target_pixel_values_hist=target_pixel_values_hist_list,
            dataset_name=dataset_name,
            actions=action,
            lang=instruction,
        )

    def _get_sequences(
        self, idx: int, window_size: int, head: bool = False
    ) -> dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_rgb_obs = self.process_rgb(
            episode, self.observation_space, self.transforms
        )
        seq_acts = process_actions(episode, 'actions', self.transforms)

        seq_lang = self.process_language(
            episode, self.transforms, self.with_lang
        )
        seq_dict = {
            **seq_rgb_obs,
            **seq_acts,
            **seq_lang,
        }  # type:ignore
        seq_dict['idx'] = idx  # type:ignore

        return seq_dict

    def _load_episode(
        self, idx: int, window_size: int
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = (
                self.min_window_size + len(self.episode_lookup) - idx - 1
            )
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size,
                (self.min_window_size + steps_to_next_episode - 1),
            )
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(
                idx, self.min_window_size, max_window
            )
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence['actions'])

    def _pad_sequence(
        self, seq: dict, pad_size: int, head: bool = False
    ) -> dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        # seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        # seq.update(
        #     {
        #         "rgb_obs": {
        #             k: self._pad_with_repetition(v, pad_size, head)
        #             for k, v in seq["rgb_obs"].items()
        #         }
        #     }
        # )
        # seq.update(
        #     {
        #         "depth_obs": {
        #             k: self._pad_with_repetition(v, pad_size, head)
        #             for k, v in seq["depth_obs"].items()
        #         }
        #     }
        # )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq['actions'], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update(
                    {
                        'actions': self._pad_with_repetition(
                            seq['actions'], pad_size, head
                        )
                    }
                )
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq['actions'], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(
                            seq['actions'][..., :-1], pad_size, head
                        ),
                        self._pad_with_repetition(
                            seq['actions'][..., -1:], pad_size, head
                        ),
                    ],
                    dim=-1,
                )
            seq.update({'actions': seq_acts})
        seq.update(
            {
                'state_info': {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq['state_info'].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(
        input_tensor: torch.Tensor, pad_size: int, head: bool = False
    ) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0),
                repeats=pad_size,
                dim=0,
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0),
                repeats=pad_size,
                dim=0,
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(
        input_tensor: torch.Tensor, pad_size: int, head: bool = False
    ) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: dict, idx: int) -> dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx]
            < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info['use_for_aux_lang_loss'] = use_for_aux_lang_loss
        return info


class DebugDataset(Dataset):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__()

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, index):
        window_size = 8
        rgb = torch.randn(window_size, 3, 200, 200)
        gripper = torch.randn(window_size, 84, 84)
        state = torch.randn(window_size, 15)


class DiskR2RDataset(BaseR2RDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = 'npz',
        pretrain: bool = False,
        partial_data=False,
        imagenet_norm=True,
        padding_sequence=False,
        padding_aug=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.partial_data = partial_data
        if self.save_format == 'pkl':
            self.load_file = load_pkl
        elif self.save_format == 'npz':
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.imagenet_norm = imagenet_norm
        self.padding_sequence = padding_sequence
        self.padding_aug = padding_aug

        if self.with_lang:
            (
                self.episode_lookup,
                self.episode_lookup_end_idx,
                self.episode_start_idx,
                self.lang_lookup,
                self.lang_ann,
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            (
                self.episode_lookup,
                self.episode_lookup_end_idx,
                self.episode_start_idx,
            ) = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f'{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}'
        )

    def _load_episode(
        self, idx: int, window_size: int
    ) -> dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """

        if self.padding_sequence:
            start_idx = self.episode_lookup[idx]
            end_idx = self.episode_lookup_end_idx[idx]
            extra_frame_num = self.max_window_size - self.min_window_size
        else:
            start_idx = self.episode_lookup[idx]
            end_idx = (
                start_idx + window_size
            )  # * self.sampling_step + self.sampling_step

        keys = list(chain(*self.observation_space.values()))
        keys.remove('language')
        # keys.append("scene_obs")

        # try:
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx, self.sampling_step)
        ]
        len_episodes = len(episodes)
        if self.padding_sequence and len_episodes < window_size:
            # print("**", start_idx, self.episode_start_idx[idx], self.min_window_size)
            if (
                self.min_window_size < self.max_window_size
                and start_idx
                < self.episode_start_idx[idx]
                + (self.max_window_size - self.min_window_size + 1)
            ):
                pad_idx = list(range(start_idx))[-extra_frame_num:]
                if len(pad_idx) < extra_frame_num:
                    pad_idx = [self.episode_start_idx[idx]] * (
                        extra_frame_num - len(pad_idx)
                    ) + pad_idx
                pad = [
                    self.load_file(self._get_episode_name(pad_idx[i]))
                    for i in range(window_size - len_episodes)
                ]
                # TODO: action->0!!
                episodes = pad + episodes
                seq_idx = pad_idx + list(
                    range(start_idx, end_idx, self.sampling_step)
                )
                # print("seq_idx:", seq_idx)
            else:
                episodes += [
                    self.load_file(self._get_episode_name(end_idx - 1))
                    for _ in range(window_size - len_episodes)
                ]

        assert len(episodes) == window_size

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        # print(start_idx, self.episode_start_idx[idx], self.min_window_size)
        if start_idx < self.episode_start_idx[idx] + self.min_window_size:
            for i in range(window_size - len_episodes):
                if seq_idx[i + 1] == seq_idx[i]:
                    episode['actions'][i] = np.zeros_like(
                        episode['actions'][i]
                    )

        if self.with_lang:
            episode['language'] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(
                    self.enrich_lang[task] + [episode['language']]
                )
                episode['language'] = enrich_lang
        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        episode_lookup_end_idx = []
        episode_start_idx = []

        try:
            print(
                'trying to load lang data from: ',
                abs_datasets_dir / self.lang_folder / 'auto_lang_ann.npy',
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / 'auto_lang_ann.npy',
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                'Exception, trying to load lang data from: ',
                abs_datasets_dir / 'auto_lang_ann.npy',
            )
            lang_data = np.load(
                abs_datasets_dir / 'auto_lang_ann.npy', allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data['indx']  # each of them are 64
        lang_ann = lang_data['language'][
            'ann'
        ]  # length total number of annotations
        lang_lookup = []

        total_eps = len(ep_start_end_ids)

        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx
                    + 1
                    - self.min_window_size
                    - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0

            # max_extra_frame_num = self.max_window_size - self.min_window_size
            # min_extra_frame_num = 1
            extra_frame_num = self.max_window_size - self.min_window_size
            if self.padding_sequence:
                if self.min_window_size != self.max_window_size:
                    for idx in range(start_idx, start_idx + extra_frame_num):
                        if cnt % self.skip_frames == 0:
                            lang_lookup.append(i)
                            episode_lookup.append(idx)
                            episode_lookup_end_idx.append(
                                idx + self.min_window_size
                            )
                            episode_start_idx.append(start_idx)
                        cnt += 1

                    for idx in range(start_idx, end_idx - extra_frame_num):
                        if cnt % self.skip_frames == 0:
                            if (
                                self.padding_aug
                                and end_idx + 1 < idx + self.max_window_size
                            ):
                                for i in range(5):
                                    lang_lookup.append(i)
                                    episode_lookup.append(idx)
                                    episode_lookup_end_idx.append(
                                        min(
                                            idx + self.max_window_size,
                                            end_idx + 1,
                                        )
                                    )
                                    episode_start_idx.append(start_idx)
                            else:
                                lang_lookup.append(i)
                                episode_lookup.append(idx)
                                episode_lookup_end_idx.append(
                                    min(
                                        idx + self.max_window_size, end_idx + 1
                                    )
                                )
                                episode_start_idx.append(start_idx)
                        cnt += 1
                elif self.min_window_size == 1 and self.max_window_size == 1:
                    for idx in range(start_idx, end_idx + 1):
                        if cnt % self.skip_frames == 0:
                            lang_lookup.append(i)
                            episode_lookup.append(idx)
                            episode_lookup_end_idx.append(
                                min(idx + self.max_window_size, end_idx + 1)
                            )
                            episode_start_idx.append(start_idx)
                        cnt += 1
                else:
                    for idx in range(start_idx, end_idx):
                        if cnt % self.skip_frames == 0:
                            lang_lookup.append(i)
                            episode_lookup.append(idx)
                            episode_lookup_end_idx.append(
                                min(idx + self.max_window_size, end_idx + 1)
                            )
                            episode_start_idx.append(start_idx)
                        cnt += 1

            else:
                for idx in range(
                    start_idx, end_idx + 1 - self.min_window_size
                ):
                    if cnt % self.skip_frames == 0:
                        lang_lookup.append(i)
                        episode_lookup.append(idx)
                    cnt += 1

        return (
            np.array(episode_lookup),
            np.array(episode_lookup_end_idx),
            np.array(episode_start_idx),
            lang_lookup,
            lang_ann,
        )

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        episode_lookup_end_idx = []
        episode_start_idx = []

        ep_start_end_ids = np.load(abs_datasets_dir / 'ep_start_end_ids.npy')
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size

            if self.padding_sequence:
                for idx in range(start_idx, start_idx + self.min_window_size):
                    episode_lookup.append(idx)
                    episode_lookup_end_idx.append(idx + self.min_window_size)
                    episode_start_idx.append(start_idx)

                for idx in range(start_idx, end_idx - extra_frame_num):
                    episode_lookup.append(idx)
                    episode_lookup_end_idx.append(
                        min(idx + window_size, end_idx + 1)
                    )
                    episode_start_idx.append(start_idx)

            else:
                for idx in range(
                    start_idx, end_idx + 1 - self.min_window_size
                ):
                    episode_lookup.append(idx)

        return (
            np.array(episode_lookup),
            np.array(episode_lookup_end_idx),
            np.array(episode_start_idx),
        )


def load_pkl(filename: Path) -> dict[str, np.ndarray]:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_npz(filename: Path) -> dict[str, np.ndarray]:
    return np.load(filename.as_posix())
