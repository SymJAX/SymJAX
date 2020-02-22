import jax.numpy as jnp
import jax.lax as jla
from .base import Tensor, Op, _add_method, jax_wrap
from . import ops_math


## getitem operator
getitem = jax_wrap(jnp.lax_numpy._rewriting_take)

_add_method(Tensor)(getitem, '__getitem__')

## overloading the basic arithmetic operators
_add_method(Tensor)(ops_math.add, '__add__')
_add_method(Tensor)(ops_math.add, '__radd__')
_add_method(Tensor)(ops_math.multiply, '__mul__')
_add_method(Tensor)(ops_math.multiply, '__rmul__')
_add_method(Tensor)(ops_math.true_divide, '__truediv__')
_add_method(Tensor)(lambda a, b: ops_math.true_divide(b, a), '__rtruediv__')
_add_method(Tensor)(ops_math.subtract, '__sub__')
_add_method(Tensor)(lambda a, b: ops_math.subtract(b,a), '__rsub__')
_add_method(Tensor)(ops_math.power, '__pow__')
_add_method(Tensor)(lambda a: 0 - a, '__neg__')

## overloading comparison operators
_add_method(Tensor)(ops_math.less, '__lt__')
_add_method(Tensor)(ops_math.greater_equal, '__rlt__')
_add_method(Tensor)(ops_math.greater, '__gt__')
_add_method(Tensor)(ops_math.less_equal, '__rgt__')
_add_method(Tensor)(ops_math.greater_equal, '__ge__')
_add_method(Tensor)(ops_math.less, '__rge__')
_add_method(Tensor)(ops_math.less_equal, '__le__')
_add_method(Tensor)(ops_math.greater, '__rle__')

## additional operators
_add_method(Tensor)(ops_math.sum, 'sum')
_add_method(Tensor)(ops_math.prod, 'prod')
_add_method(Tensor)(ops_math.mean, 'mean')
_add_method(Tensor)(ops_math.max, 'max')
_add_method(Tensor)(ops_math.min, 'min')
_add_method(Tensor)(ops_math.std, 'std')
_add_method(Tensor)(ops_math.var, 'var')
_add_method(Tensor)(ops_math.argmax, 'argmax')
_add_method(Tensor)(ops_math.argmin, 'argmin')




## additional operators
_add_method(Tensor)(ops_math.cast, 'cast')
_add_method(Tensor)(ops_math.cast, 'astype')
_add_method(Tensor)(ops_math.squeeze, 'squeeze')
_add_method(Tensor)(ops_math.flatten, 'flatten')
_add_method(Tensor)(ops_math.reshape, 'reshape')
_add_method(Tensor)(ops_math.T, 'T')
_add_method(Tensor)(ops_math.T, 'transpose')
_add_method(Tensor)(ops_math.dot, 'dot')
_add_method(Tensor)(ops_math.repeat, 'repeat')
_add_method(Tensor)(ops_math.expand_dims, 'expand_dims')
_add_method(Tensor)(ops_math.matmul, 'matmul')
_add_method(Tensor)(ops_math.round, 'round')
