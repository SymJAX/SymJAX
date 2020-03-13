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

_JNP_NAMES = [c[0] for c in inspect.getmembers(jnp, inspect.isfunction)]
_TO_SKIP = [
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
    'array2string',
    'array_equal',
    'array_equiv',
    'array_repr',
    'array_split',
    'array_str',
    'asanyarray',
    'asarray_chkfinite',
    'ascontiguousarray',
    'asfarray',
    'asfortranarray',
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


for name in _JNP_NAMES:
    if name in _TO_SKIP:
        continue
    module.__dict__.update({name: jax_wrap(jnp.__dict__[name])})

cast = jax_wrap(jla.convert_element_type)
complex = jax_wrap(jla.complex)
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

