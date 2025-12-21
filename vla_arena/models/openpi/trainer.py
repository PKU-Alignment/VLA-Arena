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
JAX training entrypoint for PI0/PI05 with multi-GPU and multi-node support.
This script mirrors the behavior of the PyTorch trainer (`trainer.py`) but runs
entirely in JAX using Flax NNX and your existing config/data pipeline.

Usage
Single GPU:
  python trainer_jax.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python trainer_jax.py debug --exp_name jax_test
  python trainer_jax.py debug --exp_name jax_test --resume  # Resume from latest checkpoint
Multi-GPU/Multi-Node:
  python trainer_jax.py <config_name> --exp_name <run_name>
  Example:
  python trainer_jax.py pi0_aloha_sim --exp_name jax_test
  python trainer_jax.py pi0_aloha_sim --exp_name jax_test --resume

With YAML config:
  python trainer_jax.py --config <path_to_config.yaml>
"""

import dataclasses
import functools
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import yaml
from flax.training import common_utils


# Add openpi src directory to Python path if needed
_openpi_src = Path(__file__).parent / 'src'
if str(_openpi_src) not in sys.path:
    sys.path.insert(0, str(_openpi_src))

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        'DEBUG': 'D',
        'INFO': 'I',
        'WARNING': 'W',
        'ERROR': 'E',
        'CRITICAL': 'C',
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(
                record.levelname, record.levelname
            )
            return super().format(record)

    formatter = CustomFormatter(
        fmt='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig, *, resuming: bool, enabled: bool = True
):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode='disabled')
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f'Checkpoint directory {ckpt_dir} does not exist.'
        )

    if resuming:
        run_id = (ckpt_dir / 'wandb_id.txt').read_text().strip()
        wandb.init(id=run_id, resume='must', project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / 'wandb_id.txt').write_text(wandb.run.id)


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape,
        got=loaded_params,
        check_shapes=True,
        check_dtypes=True,
    )

    # Remove jax.ShapeDtypeStruct from the loaded params
    import flax.traverse_util as traverse_util

    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    """Initialize training state."""
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict()
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        chunked_loss = model.compute_loss(
            rng, observation, actions, train=True
        )
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old
                + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(
                    '.*/(bias|scale|pos_embedding|input_embedding)'
                )
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        'loss': loss,
        'grad_norm': optax.global_norm(grads),
        'param_norm': optax.global_norm(kernel_params),
    }
    return new_state, info


def train_loop(config: _config.TrainConfig):
    """Main training loop."""
    init_logging()
    is_main = jax.process_index() == 0

    if is_main:
        logging.info(
            f'Running on: {platform.node()} | world_size={jax.process_count()}'
        )
        logging.info(
            f'Training config: batch_size={config.batch_size}, num_train_steps={config.num_train_steps}'
        )
        logging.info(f'LR schedule: {type(config.lr_schedule).__name__}')
        logging.info(f'Optimizer: {type(config.optimizer).__name__}')
        logging.info(f'EMA decay: {config.ema_decay}')

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f'Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}.'
        )

    jax.config.update(
        'jax_compilation_cache_dir',
        str(epath.Path('~/.cache/jax').expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)

    if is_main:
        logging.info(
            f'Initialized data loader:\n{training_utils.array_tree_to_info(batch)}'
        )

    # Log images from first batch to sanity check.
    if is_main and config.wandb_enabled and not resuming:
        images_to_log = [
            wandb.Image(
                np.concatenate(
                    [np.array(img[i]) for img in batch[0].images.values()],
                    axis=1,
                )
            )
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({'camera_views': images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)

    if is_main:
        logging.info(
            f'Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}'
        )

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager, train_state, data_loader
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(
            replicated_sharding,
            train_state_sharding,
            data_sharding,
        ),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = (
        tqdm.tqdm(
            range(start_step, config.num_train_steps),
            initial=start_step,
            total=config.num_train_steps,
            dynamic_ncols=True,
        )
        if is_main
        else None
    )

    infos = []
    start_time = None
    for step in range(start_step, config.num_train_steps):
        if step == start_step:
            start_time = jax.device_get(
                jax.block_until_ready(jax.numpy.array(jax.device_count()))
            )
            if is_main:
                import time

                start_time = time.time()

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if is_main and (step % config.log_interval == 0):
            import time

            elapsed = time.time() - start_time if start_time else 0

            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(
                jax.tree.map(jnp.mean, stacked_infos)
            )
            info_str = ', '.join(
                f'{k}={v:.4f}' for k, v in reduced_info.items()
            )

            logging.info(f'step={step} {info_str} time={elapsed:.1f}s')

            # Log to wandb
            if config.wandb_enabled:
                log_payload = dict(reduced_info)
                log_payload['step'] = step
                log_payload['time_per_step'] = (
                    elapsed / config.log_interval
                    if config.log_interval > 0
                    else 0
                )
                wandb.log(log_payload, step=step)

            if start_time:
                start_time = time.time()
            infos = []

        batch = next(data_iter)

        if (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            if is_main:
                _checkpoints.save_state(
                    checkpoint_manager, train_state, data_loader, step
                )
                logging.info(f'Saved checkpoint at step {step}')

        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            if infos:
                latest_info = infos[-1]
                pbar.set_postfix(
                    {
                        'loss': f"{latest_info['loss']:.4f}",
                        'grad_norm': f"{latest_info.get('grad_norm', 0):.2f}",
                        'step': step,
                    }
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    if is_main:
        logging.info('Waiting for checkpoint manager to finish')
    checkpoint_manager.wait_until_finished()


def main(
    config: _config.TrainConfig | str | Path | None = None, **override_kwargs
):
    """
    Main entry point for training.

    Args:
        config: Can be:
            - None: Use CLI to load config (default behavior)
            - TrainConfig: Use provided config object
            - str/Path: Path to config YAML file
        **override_kwargs: Additional keyword arguments to override config values (e.g., overwrite=True)
    """
    init_logging()

    # [Config Parsing] Handle cases where config is a path
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at: {config_path}')

        print(f'Loading configuration from {config_path}...')

        # Load YAML file
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        # Apply overrides from kwargs
        if override_kwargs:
            yaml_data.update(override_kwargs)

        # If yaml contains a config name, use it with tyro
        if isinstance(yaml_data, dict) and 'name' in yaml_data:
            config_name = yaml_data['name']

            # Recursively convert nested dict to command line args
            def dict_to_args(prefix: str, d: dict) -> list[str]:
                """Recursively convert nested dict to tyro command line args."""
                args = []
                for key, value in d.items():
                    if key == 'name':
                        continue
                    full_key = f'{prefix}.{key}' if prefix else key
                    if isinstance(value, dict):
                        # Recursively handle nested dicts
                        args.extend(dict_to_args(full_key, value))
                    elif isinstance(value, (list, tuple)):
                        # Handle lists/tuples
                        args.append(
                            f"--{full_key}={','.join(str(v) for v in value)}"
                        )
                    elif isinstance(value, bool):
                        # Handle booleans: only add flag if True
                        # For False, skip (use default) since tyro doesn't accept --key=false
                        if value:
                            args.append(f'--{full_key}')
                        # else: skip False values to use default
                    elif value is None:
                        # Skip None values
                        continue
                    else:
                        args.append(f'--{full_key}={value}')
                return args

            # Build command line args from yaml
            original_argv = sys.argv.copy()
            try:
                args_list = [config_name]  # Start with config name
                args_list.extend(dict_to_args('', yaml_data))

                # Temporarily modify sys.argv to pass args to tyro
                sys.argv = ['trainer_jax.py'] + args_list
                cfg = _config.cli()
            finally:
                # Restore original argv
                sys.argv = original_argv
        else:
            # Fallback: use CLI if yaml doesn't have expected structure
            print(
                "Warning: Config file doesn't have expected structure, falling back to CLI"
            )
            cfg = _config.cli()

        print(
            f"Config loaded successfully. Dataset: {cfg.data.repo_id if hasattr(cfg.data, 'repo_id') else 'N/A'}, Max Steps: {cfg.num_train_steps}"
        )

    elif isinstance(config, _config.TrainConfig):
        cfg = config
    elif config is None:
        # Default behavior: use CLI
        cfg = _config.cli()
    else:
        raise ValueError(
            f'Unsupported config type: {type(config)}. Expected TrainConfig, str, Path, or None.'
        )

    train_loop(cfg)


if __name__ == '__main__':
    import argparse

    # Use argparse to parse --config parameter passed by Launcher
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=None, help='Path to the config yaml file'
    )
    # This allows compatibility with other possible parameters (though currently only config is needed)
    args, unknown = parser.parse_known_args()

    # Call main with config path string (if provided)
    main(config=args.config if args.config else None)
