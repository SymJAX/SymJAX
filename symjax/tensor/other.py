import jax.numpy as jnp
import jax
import jax.lax as jla
from .base import Op, Tuple, jax_wrap
from .control_flow import cond
#import ast
#import inspect
import sys
from .index_ops import index_add
from . import numpy as snp
from .ops_nn import relu

module = sys.modules[__name__]



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
    output = (relu(x - t_left)) * slope_left\
        - relu(x - t_center) * (slope_left + slope_right)\
        + relu(x - t_right) * slope_right
    return output


def _extract_signal_patches(signal, window_length, hop=1, data_format='NCW'):
    assert not hasattr(window_length, '__len__')
    assert signal.ndim == 3
    if data_format == 'NCW':
        N = (signal.shape[2] - window_length) // hop + 1
        indices = jnp.arange(window_length) +\
            jnp.expand_dims(jnp.arange(N) * hop, 1)
        indices = jnp.reshape(indices, [1, 1, N * window_length])
        patches = jnp.take_along_axis(signal, indices, 2)
        return jnp.reshape(patches, signal.shape[:2] + (N, window_length))
    else:
        error

extract_signal_patches = jax_wrap(_extract_signal_patches, module)


def _extract_image_patches(image, window_shape, hop=1, data_format='NCHW',
                           mode='valid'):
    if mode == 'same':
        p1 = (window_shape[0] - 1)
        p2 = (window_shape[1] - 1)
        image = jnp.pad(image, [(0, 0), (0, 0), (p1 // 2, p1 - p1 // 2),
                                (p2 // 2, p2 - p2 // 2)])
    if not hasattr(hop, '__len__'):
        hop = (hop, hop)

    if image.ndim == 3:
        image = image[:, None, :, :]
        input_dim = 3
    elif image.ndim == 2:
        image = image[None, None, :, :]
        input_dim = 2
    else:
        input_dim = 4
    if data_format == 'NCHW':

        # compute the number of windows in both dimensions
        N = ((image.shape[2] - window_shape[0]) // hop[0] + 1,
             (image.shape[3] - window_shape[1]) // hop[1] + 1)

        # compute the base indices of a 2d patch
        total = 1
        for i in window_shape:
            total *= i
        patch = jnp.arange(total).reshape(window_shape)
        offset = jnp.expand_dims(jnp.arange(window_shape[0]), 1)
        patch_indices = patch + offset * (image.shape[3] - window_shape[1])

        # create all the shifted versions of it
        ver_shifts = jnp.reshape(
            jnp.arange(
                N[0]) * hop[0] * image.shape[3], (-1, 1, 1, 1))
        hor_shifts = jnp.reshape(jnp.arange(N[1]) * hop[1], (-1, 1, 1))
        all_cols = patch_indices + jnp.reshape(jnp.arange(N[1]) * hop[1],
                                               (-1, 1, 1))
        indices = patch_indices + ver_shifts + hor_shifts

        # now extract shape (1, 1, H'W'a'b')
        flat_indices = jnp.reshape(indices, [1, 1, -1])
        # shape is now (N, C, W*H)
        flat_image = jnp.reshape(
            image, (image.shape[0], image.shape[1], -1))
        # shape is now (N, C)
        patches = jnp.take_along_axis(flat_image, flat_indices, 2)
        patches = jnp.reshape(patches, image.shape[:2] + N + tuple(window_shape))
        if input_dim == 2:
            return patches[0, 0]
        elif input_dim == 3:
            return patches[0]
        else:
            return patches
    else:
        error

extract_image_patches = jax_wrap(_extract_image_patches)



def _one_hot(i, N, dtype='float32'):
    """Create a one-hot encoding of x of size k."""
    if not hasattr(i, 'shape'):
        s = ()
    else:
        s = i.shape

    if len(s) == 0:
        z = jnp.zeros((N,), dtype)
        return jax.ops.index_add(z, i, 1)
    else:
        return (i[:, None] == jnp.arange(N)).astype(dtype)

one_hot = jax_wrap(_one_hot)



stop_gradient = jax_wrap(jla.stop_gradient)
dynamic_slice_in_dim = jax_wrap(jla.dynamic_slice_in_dim)
dynamic_slice = jax_wrap(jla.dynamic_slice)
