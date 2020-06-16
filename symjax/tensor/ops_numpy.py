import inspect
import sys

import jax.lax as jla
import jax.numpy as jnp

from .base import jax_wrap

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
    'array_equiv',
    'array_repr',
    'array_split',
    'array_str',
    'asanyarray',
    'asarray_chkfinite',
    'ascontiguousarray',
    'asfarray',
    'asfortranarray',
    'copy',
    'copyto',
    'custom_tra,nsforms',
    'delete',
    'deprecate',
    'device_put',
    'digitize',
    'disp',
    'function',
    'func',
    'find_common_type',
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
    'nested_iters',
    'nextafter',
    'nper',
    'npv',
    'obj2sctype',
    'packbits',
    'printoptions',
    'recfromcsv',
    'recfromtxt',
    'removechars',
    'result_type',
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
    'source',
    'spacing',
    'strtobool',
    'trapz',
    'typename',
    'union1d',
    'unique',
    'unpackbits',
    'update_numpydoc',
    'who'
]

for name in _JNP_NAMES:
    if name in _TO_SKIP:
        continue
    module.__dict__.update({name: jax_wrap(jnp.__dict__[name])})

cast = jax_wrap(jla.convert_element_type)
complex = jax_wrap(jla.complex)
range = module.__dict__['arange']
T = module.__dict__['transpose']


def flatten(input):
    return module.__dict__['reshape'](input, (-1,))


def flatten2d(input):
    assert input.ndim > 1
    if input.ndim == 2:
        return input
    return reshape(input, (input.shape[0], -1))


def logsumexp(x, axis):
    x_max = module.__dict__['stop_gradient'](x.max(axis, keepdims=True))
    return module.__dict__['log'](
        (module.__dict__['exp'](x - x_max)).sum(axis)) + \
           module.__dict__['squeeze'](x_max)
