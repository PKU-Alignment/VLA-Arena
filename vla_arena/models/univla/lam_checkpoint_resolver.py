"""Utilities for resolving and loading UniVLA LAM checkpoints."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


_CHECKPOINT_SUFFIXES = ('.ckpt', '.pt')
_STAGE_PATTERN = re.compile(r'stage[_-]?(\d+)', re.IGNORECASE)
_STEP_PATTERN = re.compile(r'step[=_-]?(\d+)', re.IGNORECASE)
_EPOCH_PATTERN = re.compile(r'epoch[=_-]?(\d+)', re.IGNORECASE)


@dataclass(frozen=True)
class ResolvedLamCheckpoint:
    resolved_path: str
    source: str
    effective_lam_path: str
    selected_checkpoint: str | None = None
    env_overridden: bool = False


def _extract_number(pattern: re.Pattern[str], text: str) -> int:
    match = pattern.search(text)
    return int(match.group(1)) if match else -1


def _is_checkpoint_file(path: str) -> bool:
    return path.lower().endswith(_CHECKPOINT_SUFFIXES)


def _checkpoint_score(path: str) -> tuple[int, int, int, int, str]:
    stage_num = _extract_number(_STAGE_PATTERN, path)
    step_num = _extract_number(_STEP_PATTERN, path)
    epoch_num = _extract_number(_EPOCH_PATTERN, path)
    ext_priority = 1 if path.lower().endswith('.ckpt') else 0
    return stage_num, step_num, epoch_num, ext_priority, path


def _choose_checkpoint(candidates: list[str], context: str) -> str:
    if not candidates:
        raise FileNotFoundError(
            f'No checkpoint files (*.ckpt/*.pt) found in {context}. '
            'Set `lam_ckpt_file` to specify an exact checkpoint file.'
        )
    return max(candidates, key=_checkpoint_score)


def _looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    if not value or not all(isinstance(k, str) for k in value.keys()):
        return False
    return any(hasattr(v, 'shape') for v in value.values())


def extract_lam_state_dict(ckpt_obj: Any) -> Mapping[str, Any]:
    """Extract state_dict from supported checkpoint formats."""
    if not isinstance(ckpt_obj, Mapping):
        raise ValueError(
            'LAM checkpoint must be a mapping object or contain a `state_dict` mapping.'
        )

    if 'state_dict' in ckpt_obj:
        state_dict = ckpt_obj['state_dict']
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                'LAM checkpoint key `state_dict` exists but is not a mapping.'
            )
        if not _looks_like_state_dict(state_dict):
            raise ValueError(
                'LAM checkpoint `state_dict` does not look like model weights.'
            )
        return state_dict

    if _looks_like_state_dict(ckpt_obj):
        return ckpt_obj

    raise ValueError(
        'Unsupported LAM checkpoint format. Expected `{"state_dict": ...}` '
        'or a plain state_dict mapping.'
    )


def resolve_lam_checkpoint(
    lam_path: str,
    lam_ckpt_file: str | None = None,
    env_var: str = 'UNIVLA_LAM_PATH',
) -> ResolvedLamCheckpoint:
    env_value = os.getenv(env_var)
    has_env_override = bool(env_value and env_value.strip())
    effective_lam_path = env_value.strip() if has_env_override else lam_path

    path_candidate = Path(effective_lam_path).expanduser()

    if path_candidate.is_file():
        return ResolvedLamCheckpoint(
            resolved_path=str(path_candidate),
            source='local_file',
            effective_lam_path=effective_lam_path,
            selected_checkpoint=str(path_candidate),
            env_overridden=has_env_override,
        )

    if lam_ckpt_file and Path(lam_ckpt_file).is_absolute():
        raise ValueError(
            '`lam_ckpt_file` must be a relative path, not an absolute path.'
        )

    if path_candidate.is_dir():
        if lam_ckpt_file:
            selected_path = path_candidate / lam_ckpt_file
            if not selected_path.is_file():
                raise FileNotFoundError(
                    f'LAM checkpoint file not found: {selected_path}'
                )
        else:
            local_candidates = [
                str(path)
                for path in path_candidate.rglob('*')
                if path.is_file() and _is_checkpoint_file(path.name)
            ]
            selected_path = Path(
                _choose_checkpoint(
                    local_candidates,
                    context=f'local directory `{path_candidate}`',
                )
            )
        return ResolvedLamCheckpoint(
            resolved_path=str(selected_path),
            source='local_dir',
            effective_lam_path=effective_lam_path,
            selected_checkpoint=str(selected_path),
            env_overridden=has_env_override,
        )

    repo_id = effective_lam_path
    target_filename = lam_ckpt_file
    try:
        if target_filename is None:
            repo_files = HfApi().list_repo_files(
                repo_id=repo_id,
                repo_type='model',
            )
            checkpoint_files = [
                filename
                for filename in repo_files
                if _is_checkpoint_file(filename)
            ]
            target_filename = _choose_checkpoint(
                checkpoint_files, context=f'HF repo `{repo_id}`'
            )

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=target_filename,
            repo_type='model',
        )
    except FileNotFoundError:
        raise
    except Exception as err:
        target = (
            f'{repo_id}/{target_filename}'
            if target_filename is not None
            else repo_id
        )
        raise RuntimeError(
            f'Failed to download LAM checkpoint `{target}` from Hugging Face Hub. '
            'Please check repo/file names, run `hf auth login` if needed, '
            'or use a local checkpoint path.'
        ) from err

    return ResolvedLamCheckpoint(
        resolved_path=downloaded_path,
        source='hf_repo',
        effective_lam_path=repo_id,
        selected_checkpoint=target_filename,
        env_overridden=has_env_override,
    )


def resolve_lam_checkpoint_path(
    lam_path: str,
    lam_ckpt_file: str | None = None,
    env_var: str = 'UNIVLA_LAM_PATH',
) -> str:
    return resolve_lam_checkpoint(
        lam_path=lam_path,
        lam_ckpt_file=lam_ckpt_file,
        env_var=env_var,
    ).resolved_path
