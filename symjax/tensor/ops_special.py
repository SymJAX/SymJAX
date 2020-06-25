import inspect
import sys

import jax
import jax.lax as jla
import jax.numpy as jnp
import numpy

from .base import jax_wrap
from ..nn.ops_nn import relu

module = sys.modules[__name__]

index = jax.ops.index


def hat_1D(x, t_left, t_center, t_right):
    """hat basis function in 1-D

    Hat function, continuous piecewise linear

    Parameters
    ----------

    x: array-like
        the sampled input space

    t_left: scalar
        the position of the left knot

    t_center: scalar
        the position of the center knot

    t_right: scalar
        the position of the right knot

    Returns
    -------

    output : array
        same shape as x with applied hat function
    """
    eps = 1e-6
    slope_left = 1 / (t_center - t_left)
    slope_right = 1 / (t_right - t_center)
    output = (
        (relu(x - t_left)) * slope_left
        - relu(x - t_center) * (slope_left + slope_right)
        + relu(x - t_right) * slope_right
    )
    return output


def _extract_signal_patches(signal, window_length, hop=1, data_format="NCW"):
    assert not hasattr(window_length, "__len__")
    assert signal.ndim == 3
    if data_format == "NCW":
        N = (signal.shape[2] - window_length) // hop + 1
        indices = jnp.arange(window_length) + jnp.expand_dims(jnp.arange(N) * hop, 1)
        indices = jnp.reshape(indices, [1, 1, N * window_length])
        patches = jnp.take_along_axis(signal, indices, 2)
        return jnp.reshape(patches, signal.shape[:2] + (N, window_length))
    else:
        error


extract_signal_patches = jax_wrap(_extract_signal_patches, module)


def _extract_image_patches(
    image, window_shape, hop=1, data_format="NCHW", mode="valid"
):
    if mode == "same":
        p1 = window_shape[0] - 1
        p2 = window_shape[1] - 1
        image = jnp.pad(
            image, [(0, 0), (0, 0), (p1 // 2, p1 - p1 // 2), (p2 // 2, p2 - p2 // 2)]
        )
    if not hasattr(hop, "__len__"):
        hop = (hop, hop)
    if data_format == "NCHW":

        # compute the number of windows in both dimensions
        N = (
            (image.shape[2] - window_shape[0]) // hop[0] + 1,
            (image.shape[3] - window_shape[1]) // hop[1] + 1,
        )

        # compute the base indices of a 2d patch
        patch = jnp.arange(numpy.prod(window_shape)).reshape(window_shape)
        offset = jnp.expand_dims(jnp.arange(window_shape[0]), 1)
        patch_indices = patch + offset * (image.shape[3] - window_shape[1])

        # create all the shifted versions of it
        ver_shifts = jnp.reshape(
            jnp.arange(N[0]) * hop[0] * image.shape[3], (-1, 1, 1, 1)
        )
        hor_shifts = jnp.reshape(jnp.arange(N[1]) * hop[1], (-1, 1, 1))
        all_cols = patch_indices + jnp.reshape(jnp.arange(N[1]) * hop[1], (-1, 1, 1))
        indices = patch_indices + ver_shifts + hor_shifts

        # now extract shape (1, 1, H'W'a'b')
        flat_indices = jnp.reshape(indices, [1, 1, -1])
        # shape is now (N, C, W*H)
        flat_image = jnp.reshape(image, (image.shape[0], image.shape[1], -1))
        # shape is now (N, C)
        patches = jnp.take_along_axis(flat_image, flat_indices, 2)
        return jnp.reshape(patches, image.shape[:2] + N + tuple(window_shape))
    else:
        error


extract_image_patches = jax_wrap(_extract_image_patches)


def _add_n(args):
    start = args[0]
    for arg in args:
        start = jnp.add(start, arg)
    return start


add_n = jax_wrap(_add_n)


def one_hot(i, N, dtype="float32"):
    """Create a one-hot encoding of x of size k."""
    if hasattr(i, "shape"):
        return (x[:, None] == arange(k)).astype(dtype)
    else:
        z = zeros(N, dtype)
        print(i, N)
        return index_add(z, i, 1)


for name in ["index_update", "index_min", "index_add", "index_max"]:
    module.__dict__.update({name: jax_wrap(jax.ops.__dict__[name])})

stop_gradient = jax_wrap(jla.stop_gradient)
dynamic_slice_in_dim = jax_wrap(jla.dynamic_slice_in_dim)
dynamic_slice = jax_wrap(jla.dynamic_slice)
index = jax.ops.index


module = sys.modules[__name__]

_NAMES = [c[0] for c in inspect.getmembers(jax.scipy.special, inspect.isfunction)]


for name in _NAMES:
    if name[0] == "_":
        continue
    module.__dict__.update({name: jax_wrap(jax.scipy.special.__dict__[name])})
