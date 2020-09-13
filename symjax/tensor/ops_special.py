import inspect
import sys

import jax
import jax.lax as jla
import jax.numpy as jnp
import numpy

from . import ops_numpy as T
from .base import jax_wrap
from ..nn.ops_nn import relu

module = sys.modules[__name__]

index = jax.ops.index


def _add_n(args):
    start = args[0]
    for arg in args:
        start = jnp.add(start, arg)
    return start


add_n = jax_wrap(_add_n)


def one_hot(i, N, dtype="float32"):
    """Create a one-hot encoding of x of size k."""
    if not hasattr(i, "shape"):
        i = T.array(i)

    if i.ndim:
        return T.equal(i[:, None], T.arange(N)).astype(dtype)
    else:
        z = T.zeros(N, dtype)
        return index_add(z, i, 1)


for name in [
    "index_update",
    "index_min",
    "index_add",
    "index_max",
]:
    module.__dict__.update({name: jax_wrap(jax.ops.__dict__[name])})

stop_gradient = jax_wrap(jla.stop_gradient)
dynamic_slice_in_dim = jax_wrap(jla.dynamic_slice_in_dim)
dynamic_slice = jax_wrap(jla.dynamic_slice)
index = jax.ops.index

index_take = jax_wrap(jax.lax.index_take)
index_in_dim = jax_wrap(jax.lax.index_in_dim)
dynamic_index_in_dim = jax_wrap(jax.lax.dynamic_index_in_dim)


module = sys.modules[__name__]

_NAMES = [c[0] for c in inspect.getmembers(jax.scipy.special, inspect.isfunction)]


for name in _NAMES:
    if name[0] == "_":
        continue
    module.__dict__.update({name: jax_wrap(jax.scipy.special.__dict__[name])})


def reshape_weight_to_matrix(self, weight, dim=1):

    if dim != 0:
        # permute dim to front
        weight_t = weight.permute(dim, *[d for d in range(weight.ndim) if d != dim])
    else:
        weight_t = weight

    return weight_t.flatten2d()
