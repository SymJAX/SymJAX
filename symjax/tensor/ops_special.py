import inspect
import sys

import jax
import jax.lax as jla
import jax.numpy as jnp

from . import ops_numpy as T
from .base import jax_wrap

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


# methods from jax.ops
for name in [
    "index_update",
    "index_min",
    "index_add",
    "index_max",
]:
    module.__dict__.update({name: jax_wrap(jax.ops.__dict__[name])})


# methods from jax lax
for name in [
    "stop_gradient",
    "dynamic_slice_in_dim",
    "dynamic_slice",
    "rsqrt",
    "index_take",
    "index_in_dim",
    "dynamic_index_in_dim",
]:
    module.__dict__.update({name: jax_wrap(jax.lax.__dict__[name])})

# stop_gradient = jax_wrap(jla.stop_gradient)
# dynamic_slice_in_dim = jax_wrap(jla.dynamic_slice_in_dim)
# dynamic_slice = jax_wrap(jla.dynamic_slice)
# rsqrt = jax_wrap(jla.rsqrt)

# index_take = jax_wrap(jax.lax.index_take)
# index_in_dim = jax_wrap(jax.lax.index_in_dim)
# dynamic_index_in_dim = jax_wrap(jax.lax.dynamic_index_in_dim)


# from jax.scipy.special
_NAMES = inspect.getmembers(jax.scipy.special, callable)  # inspect.isfunction)


for name, func in _NAMES:
    if name[0] == "_":
        continue
    module.__dict__.update({name: jax_wrap(func)})

module.__dict__["sigmoid"] = module.__dict__["expit"]


def reshape_weight_to_matrix(self, weight, dim=1):

    if dim != 0:
        # permute dim to front
        weight_t = weight.permute(dim, *[d for d in range(weight.ndim) if d != dim])
    else:
        weight_t = weight

    return weight_t.flatten2d()


def dimshuffle(tensor, pattern):
    """Reorder the dimensions of this variable, optionally inserting
    broadcasted dimensions.

    Parameters
    ----------
    tensor: Tensor

    pattern: list of int and str
        List/tuple of int mixed with 'x' for broadcastable dimensions.
    Examples
    --------
    For example, to create a 3D view of a [2D] matrix, call
    ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
    middle dimension is an implicit broadcasted dimension.  To do the same
    thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.
    Notes
    -----
    This function supports the pattern passed as a tuple, or as a
    variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
    to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
    mixed with 'x' characters).
    """

    # first get the transpose ordering
    transpose_pattern = [p for p in pattern if type(p) == int]
    tensor_T = T.transpose(tensor, transpose_pattern)

    # now take care of the expand_dims
    shapes = tensor_T.shape.__iter__()
    expand_shape = [shapes.__next__() if type(t) == int else 1 for t in pattern]
    return tensor_T.reshape(expand_shape)
