from .base import Tensor, _add_method
import inspect
import sys

import jax.lax as jla
import jax.numpy as jnp

from .base import jax_wrap

module = sys.modules[__name__]

_JNP_NAMES = [c[0] for c in inspect.getmembers(jnp, inspect.isfunction)]
_TO_SKIP = [
    "<lambda>",
    "blackman",
    "bartlett",
    "hamming",
    "hanning",
    "kaiser",
    "add_docstring",
    "add_newdoc",
    "alen",
    "apply_along_axis",
    "apply_over_axes",
    "array2string",
    "array_equiv",
    "array_repr",
    "array_split",
    "array_str",
    "asanyarray",
    "asarray_chkfinite",
    "ascontiguousarray",
    "asfarray",
    "asfortranarray",
    "copy",
    "copyto",
    "custom_tra,nsforms",
    "delete",
    "deprecate",
    "device_put",
    "digitize",
    "disp",
    "function",
    "func",
    "find_common_type",
    "format_float_positional",
    "format_float_scientific",
    "frexp",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromregex",
    "fromstring",
    "fv",
    "genfromtxt",
    "get_array_wrap",
    "get_include",
    "get_module_functions",
    "get_printoptions",
    "getbufsize",
    "geterr",
    "geterrcall",
    "geterrobj",
    "gradient",
    "int_asbuffer",
    "is_busday",
    "isrealobj",
    "issctype",
    "issubclass_",
    "issubdtype",
    "issubsctype",
    "iterable",
    "jit",
    "load",
    "loads",
    "loadtxt",
    "lookfor",
    "mafromtxt",
    "maximum_sctype",
    "may_share_memory",
    "mintypecode",
    "ndfromtxt",
    "nested_iters",
    "nextafter",
    "nper",
    "npv",
    "obj2sctype",
    "packbits",
    "printoptions",
    "recfromcsv",
    "recfromtxt",
    "removechars",
    "result_type",
    "safe_eval",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "sctype2char",
    "searchsorted",
    "select",
    "shape",
    "shares_memory",
    "show",
    "size",
    "source",
    "spacing",
    "strtobool",
    "trapz",
    "typename",
    "union1d",
    "unique",
    "unpackbits",
    "update_numpydoc",
    "who",
]

for name in _JNP_NAMES:
    if name in _TO_SKIP:
        continue
    module.__dict__.update({name: jax_wrap(jnp.__dict__[name])})

for name in [
    "issubsctype",
    "issubdtype",
    "nextafter",
    "result_type",
    "select",
    "isneginf",
    "isposinf",
    "logaddexp",
    "logaddexp2",
]:
    module.__dict__.update({name: jax_wrap(jnp.__dict__[name])})


cast = jax_wrap(jla.convert_element_type)
complex = jax_wrap(jla.complex)
range = module.__dict__["arange"]


def flatten(input):
    return module.__dict__["reshape"](input, (-1,))


def flatten2d(input):
    assert input.ndim > 1
    if input.ndim == 2:
        return input
    return reshape(input, (input.shape[0], -1))


################
getitem = jax_wrap(jnp.lax_numpy._rewriting_take)


_add_method(Tensor)(getitem, "__getitem__")

# overloading the basic arithmetic operators
_add_method(Tensor)(module.__dict__["add"], "__add__")
_add_method(Tensor)(module.__dict__["add"], "__radd__")
_add_method(Tensor)(module.__dict__["multiply"], "__mul__")
_add_method(Tensor)(module.__dict__["multiply"], "__rmul__")
_add_method(Tensor)(module.__dict__["true_divide"], "__truediv__")
_add_method(Tensor)(lambda a, b: module.__dict__["true_divide"](b, a), "__rtruediv__")
_add_method(Tensor)(module.__dict__["floor_divide"], "__floordiv__")
_add_method(Tensor)(lambda a, b: module.__dict__["floor_divide"](b, a), "__rfloordiv__")

_add_method(Tensor)(module.__dict__["subtract"], "__sub__")
_add_method(Tensor)(lambda a, b: module.__dict__["subtract"](b, a), "__rsub__")
_add_method(Tensor)(module.__dict__["power"], "__pow__")
_add_method(Tensor)(lambda a: 0 - a, "__neg__")

# overloading comparison operators
_add_method(Tensor)(module.__dict__["less"], "__lt__")
_add_method(Tensor)(module.__dict__["greater_equal"], "__rlt__")
_add_method(Tensor)(module.__dict__["greater"], "__gt__")
_add_method(Tensor)(module.__dict__["less_equal"], "__rgt__")
_add_method(Tensor)(module.__dict__["greater_equal"], "__ge__")
_add_method(Tensor)(module.__dict__["less"], "__rge__")
_add_method(Tensor)(module.__dict__["less_equal"], "__le__")
_add_method(Tensor)(module.__dict__["greater"], "__rle__")

# additional operators
_add_method(Tensor)(module.__dict__["sum"], "sum")
_add_method(Tensor)(module.__dict__["prod"], "prod")
_add_method(Tensor)(module.__dict__["mean"], "mean")
_add_method(Tensor)(module.__dict__["max"], "max")
_add_method(Tensor)(module.__dict__["min"], "min")
_add_method(Tensor)(module.__dict__["std"], "std")
_add_method(Tensor)(module.__dict__["var"], "var")
_add_method(Tensor)(module.__dict__["argmax"], "argmax")
_add_method(Tensor)(module.__dict__["argmin"], "argmin")

# additional operators

_add_method(Tensor)(module.__dict__["real"], "real")
_add_method(Tensor)(module.__dict__["imag"], "imag")
_add_method(Tensor)(module.__dict__["conjugate"], "conj")
_add_method(Tensor)(module.__dict__["conjugate"], "conjugate")
_add_method(Tensor)(module.__dict__["cast"], "cast")
_add_method(Tensor)(module.__dict__["cast"], "astype")
_add_method(Tensor)(module.__dict__["squeeze"], "squeeze")
_add_method(Tensor)(module.__dict__["flatten"], "flatten")
_add_method(Tensor)(module.__dict__["reshape"], "reshape")
_add_method(Tensor)(module.__dict__["dot"], "dot")
_add_method(Tensor)(module.__dict__["repeat"], "repeat")
_add_method(Tensor)(module.__dict__["expand_dims"], "expand_dims")
_add_method(Tensor)(module.__dict__["matmul"], "matmul")
_add_method(Tensor)(module.__dict__["round"], "round")
