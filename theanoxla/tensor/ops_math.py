import jax.numpy as jnp
import numpy
import jax
import jax.lax as jla
from .base import Op, Tuple, jax_wrap
from .control_flow import cond
import ast
import inspect
import sys
from .ops_activations import relu
module = sys.modules[__name__]


def hat_1D(x, t_left, t_center, t_right):
    """
    Hat function, continuous piecewise linear, such that::
        f(x) = \begin{cases}
                    0 \iff x \not \in (t_left,t_right)\\
                    1 \iff x = t_center\\
                    \frac{x - t_left}{t_center - t_left} \iff x \in (t_left, t]\\
                    \frac{x - t_center}{t_center - t_right} \iff x \in (t_left, t]
                \end{cases}
    Parameters
    ----------

    x :: array-like
        the sampled input space
    t_left :: scalar
        the position of the left knot
   t_center :: scalar
        the position of the center knot
    t_right :: scalar
        the position of the right knot

    Returns
    -------
    output :: array
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
    if data_format == 'NCW':
        N = (signal.shape[2] - window_length) // hop + 1
        indices = jnp.arange(window_length) +\
            jnp.expand_dims(jnp.arange(N) * hop, 1)
        indices = jnp.reshape(indices, [1, 1, N * window_length])
        patches = jnp.take_along_axis(signal, indices, 2)
        return jnp.reshape(patches, signal.shape[:2] + (N, window_length))
    else:
        error

extract_signal_patches = jax_wrap(_extract_signal_patches)


def _extract_image_patches(image, window_shape, hop=1, data_format='NCHW',
                           mode='valid'):
    if mode == 'same':
        p1 = (window_shape[0] - 1)
        p2 = (window_shape[1] - 1)
        image = jnp.pad(image, [(0, 0), (0, 0), (p1 // 2, p1 - p1 // 2),
                                (p2 // 2, p2 - p2 // 2)])
    if not hasattr(hop, '__len__'):
        hop = (hop, hop)
    if data_format == 'NCHW':

        # compute the number of windows in both dimensions
        N = ((image.shape[2] - window_shape[0]) // hop[0] + 1,
             (image.shape[3] - window_shape[1]) // hop[1] + 1)

        # compute the base indices of a 2d patch
        patch = jnp.arange(numpy.prod(window_shape)).reshape(window_shape)
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
        return jnp.reshape(patches,
                           image.shape[:2] + N + tuple(window_shape))
    else:
        error

extract_image_patches = jax_wrap(_extract_image_patches)


class add_n(Op):
    @staticmethod
    def fn(args):
        start = args[0]
        for arg in args:
            start = jnp.add(start, arg)
        return start


class one_hot(Op):
    @staticmethod
    def fn(i, N, dtype='float32'):
        """Create a one-hot encoding of x of size k."""
        z = jnp.zeros((N,), dtype)
        z = jax.ops.index_add(z, i, 1)
        return z


class to_one_hot(Op):
    @staticmethod
    def fn(x, k, dtype='float32'):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)


def upsample(x, factors, mode='zeros'):
    if mode == 'repeat':
        vs = [x]
        for ax, f in enumerate(factors):
            vs.append(repeat(vs[-1], f, ax))
        return vs[-1]
    elif mode == 'zeros':
        masks = [1]
        for ax, f in enumerate(factors):
            v = tile(one_hot(0, f), x.shape[ax])
            shape = (one_hot(ax, len(factors), 'int32') * (-2) + 1).get({})
            print(shape)
            masks.append(masks[-1] * v.reshape(shape))
        x_repeat = upsample(x, factors, mode='repeat')
        return x_repeat * masks[-1]
    else:
        raise ValueError('Not Implemented upsample')


JNP_NAMES = [c[0] for c in inspect.getmembers(jnp, inspect.isfunction)]
TO_SKIP = [
    '<lambda>',
    'blackman',
    'bartlett',
    'hamming',
    'hanning',
    'kaiser',
    'add_docstring',
    'add_newdoc',
    'alen',
    'apply_along_axis',
    'apply_over_axes',
    'array',
    'array2string',
    'array_equal',
    'array_equiv',
    'array_repr',
    'array_split',
    'array_str',
    'asanyarray',
    'asarray',
    'asarray_chkfinite',
    'ascontiguousarray',
    'asfarray',
    'asfortranarray',
    'asmatrix',
    'asscalar',
    'broadcast_arrays',
    'broadcast_to',
    'copy',
    'copysign',
    'copyto',
    'custom_tra,nsforms',
    'delete',
    'deprecate',
    'device_put',
    'digitize',
    'disp',
    'ediff1d',
    'function',
    'func',
    'find_common_type',
    'fix',
    'format_float_positional',
    'format_float_scientific',
    'frexp',
    'frombuffer',
    'fromfile',
    'fromfunction',
    'fromiter',
    'frompyfunc',
    'fromregex',
    'fromstring',
    'fv',
    'genfromtxt',
    'get_array_wrap',
    'get_include',
    'get_module_functions',
    'get_printoptions',
    'getbufsize',
    'geterr',
    'geterrcall',
    'geterrobj',
    'gradient',
    'histogramdd',
    'hypot',
    'int_asbuffer',
    'is_busday',
    'isrealobj',
    'issctype',
    'issubclass_',
    'issubdtype',
    'issubsctype',
    'iterable',
    'jit',
    'load',
    'loads',
    'loadtxt',
    'lookfor',
    'mafromtxt',
    'maximum_sctype',
    'may_share_memory',
    'mintypecode',
    'ndfromtxt',
    'negative',
    'nested_iters',
    'nextafter',
    'nper',
    'npv',
    'obj2sctype',
    'packbits',
    'printoptions',
    'ptp',
    'recfromcsv',
    'recfromtxt',
    'reciprocal',
    'removechars',
    'result_type',
    'right_shift',
    'rint',
    'safe_eval',
    'save',
    'savetxt',
    'savez',
    'savez_compressed',
    'sctype2char',
    'searchsorted',
    'select',
    'shape',
    'shares_memory',
    'show',
    'size',
    'sometrue',
    'source',
    'spacing',
    'strtobool',
    'trapz',
    'typename',
    'union1d',
    'unique',
    'unpackbits',
    'update_numpydoc',
    'vander',
    'who'
]


for name in JNP_NAMES:
    if name in TO_SKIP:
        continue
    module.__dict__.update({name: jax_wrap(jnp.__dict__[name])})

cast = jax_wrap(jla.convert_element_type)
complex = jax_wrap(jla.complex)
stop_gradient = jax_wrap(jla.stop_gradient)
dynamic_slice_in_dim = jax_wrap(jla.dynamic_slice_in_dim)
range = arange
T = transpose

def flatten(input):
    return reshape(input, (-1,))

def flatten2d(input):
    assert input.ndim > 1
    if input.ndim == 2:
        return input
    return reshape(input, (input.shape[0], -1))

def logsumexp(x, axis):
    x_max = stop_gradient(x.max(axis, keepdims=True))
    return log(exp(x - x_max).sum(axis)) + squeeze(x_max)
