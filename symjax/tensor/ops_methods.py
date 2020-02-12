import jax.numpy as jnp
import jax.lax as jla
from .base import Tensor, Op, add_method, jax_wrap
from . import ops_math


## getitem operator
getitem = jax_wrap(jnp.lax_numpy._rewriting_take)

add_method(Tensor)(getitem, '__getitem__')

## overloading the basic arithmetic operators
add_method(Tensor)(ops_math.add, '__add__')
add_method(Tensor)(ops_math.add, '__radd__')
add_method(Tensor)(ops_math.multiply, '__mul__')
add_method(Tensor)(ops_math.multiply, '__rmul__')
add_method(Tensor)(ops_math.true_divide, '__truediv__')
add_method(Tensor)(lambda a, b: ops_math.true_divide(b, a), '__rtruediv__')
add_method(Tensor)(ops_math.subtract, '__sub__')
add_method(Tensor)(lambda a, b: ops_math.subtract(b,a), '__rsub__')
add_method(Tensor)(ops_math.power, '__pow__')
add_method(Tensor)(lambda a: 0 - a, '__neg__')

## overloading comparison operators
#add_method(Tensor)(ops_math.equal, '__eq__')
#add_method(Tensor)(ops_math.equal, '__req__')
add_method(Tensor)(ops_math.less, '__lt__')
add_method(Tensor)(ops_math.greater_equal, '__rlt__')
add_method(Tensor)(ops_math.greater, '__gt__')
add_method(Tensor)(ops_math.less_equal, '__rgt__')
add_method(Tensor)(ops_math.greater_equal, '__ge__')
add_method(Tensor)(ops_math.less, '__rge__')
add_method(Tensor)(ops_math.less_equal, '__le__')
add_method(Tensor)(ops_math.greater, '__rle__')
#add_method(Tensor)(ops_math.not_equal, '__ne__')
#add_method(Tensor)(ops_math.not_equal, '__rne__')

## additional operators
add_method(Tensor)(ops_math.sum, 'sum')
add_method(Tensor)(ops_math.prod, 'prod')
add_method(Tensor)(ops_math.mean, 'mean')
add_method(Tensor)(ops_math.max, 'max')
add_method(Tensor)(ops_math.min, 'min')
add_method(Tensor)(ops_math.std, 'std')
add_method(Tensor)(ops_math.var, 'var')

## additional operators
add_method(Tensor)(ops_math.argmax, 'argmax')
add_method(Tensor)(ops_math.argmin, 'argmin')




## additional operators
add_method(Tensor)(ops_math.cast, 'cast')
add_method(Tensor)(ops_math.cast, 'astype')
add_method(Tensor)(ops_math.squeeze, 'squeeze')
add_method(Tensor)(ops_math.flatten, 'flatten')
add_method(Tensor)(ops_math.reshape, 'reshape')
add_method(Tensor)(ops_math.T)
add_method(Tensor)(ops_math.dot, 'dot')
add_method(Tensor)(ops_math.repeat, 'repeat')
add_method(Tensor)(ops_math.expand_dims, 'expand_dims')
add_method(Tensor)(ops_math.matmul, 'matmul')
