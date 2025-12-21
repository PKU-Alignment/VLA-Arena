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

import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
import jaxtyping._decorator
import torch
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import ArrayLike, Float, PyTree, config, jaxtyped


# patch jaxtyping to handle https://github.com/patrick-kidger/jaxtyping/issues/277.
# the problem is that custom PyTree nodes are sometimes initialized with arbitrary types (e.g., `jax.ShapeDtypeStruct`,
# `jax.Sharding`, or even <object>) due to JAX tracing operations. this patch skips typechecking when the stack trace
# contains `jax._src.tree_util`, which should only be the case during tree unflattening.
_original_check_dataclass_annotations = (
    jaxtyping._decorator._check_dataclass_annotations
)
# Redefine Array to include both JAX arrays and PyTorch tensors
Array = jax.Array | torch.Tensor


def _check_dataclass_annotations(self, typechecker):
    if not any(
        frame.frame.f_globals.get('__name__')
        in {'jax._src.tree_util', 'flax.nnx.transforms.compilation'}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = (
    _check_dataclass_annotations  # noqa: SLF001
)

KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, '...']]

T = TypeVar('T')


# runtime type-checking decorator
def typecheck(t: T) -> T:
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update('jaxtyping_disable', True)  # noqa: FBT003
    yield
    config.update('jaxtyping_disable', initial)


def check_pytree_equality(
    *,
    expected: PyTree,
    got: PyTree,
    check_shapes: bool = False,
    check_dtypes: bool = False,
):
    """Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer
    error message than if `jax.tree.map` is naively used on PyTrees with different structures.
    """

    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            'PyTrees have different structure:\n'
            + (
                '\n'.join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(
                    f'Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}'
                )

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(
                    f'Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}'
                )

        jax.tree_util.tree_map_with_path(check, expected, got)
