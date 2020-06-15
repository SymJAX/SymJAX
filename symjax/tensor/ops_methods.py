import symjax.tensor as T
from .base import Tensor, _add_method

_add_method(Tensor)(T.getitem, '__getitem__')

## overloading the basic arithmetic operators
_add_method(Tensor)(T.add, '__add__')
_add_method(Tensor)(T.add, '__radd__')
_add_method(Tensor)(T.multiply, '__mul__')
_add_method(Tensor)(T.multiply, '__rmul__')
_add_method(Tensor)(T.true_divide, '__truediv__')
_add_method(Tensor)(lambda a, b: T.true_divide(b, a), '__rtruediv__')
_add_method(Tensor)(T.floor_divide, '__floordiv__')
_add_method(Tensor)(lambda a, b: T.floor_divide(b, a), '__rfloordiv__')

_add_method(Tensor)(T.subtract, '__sub__')
_add_method(Tensor)(lambda a, b: T.subtract(b, a), '__rsub__')
_add_method(Tensor)(T.power, '__pow__')
_add_method(Tensor)(lambda a: 0 - a, '__neg__')

## overloading comparison operators
_add_method(Tensor)(T.less, '__lt__')
_add_method(Tensor)(T.greater_equal, '__rlt__')
_add_method(Tensor)(T.greater, '__gt__')
_add_method(Tensor)(T.less_equal, '__rgt__')
_add_method(Tensor)(T.greater_equal, '__ge__')
_add_method(Tensor)(T.less, '__rge__')
_add_method(Tensor)(T.less_equal, '__le__')
_add_method(Tensor)(T.greater, '__rle__')

## additional operators
_add_method(Tensor)(T.sum, 'sum')
_add_method(Tensor)(T.prod, 'prod')
_add_method(Tensor)(T.mean, 'mean')
_add_method(Tensor)(T.max, 'max')
_add_method(Tensor)(T.min, 'min')
_add_method(Tensor)(T.std, 'std')
_add_method(Tensor)(T.var, 'var')
_add_method(Tensor)(T.argmax, 'argmax')
_add_method(Tensor)(T.argmin, 'argmin')

## additional operators

_add_method(Tensor)(T.real, 'real')
_add_method(Tensor)(T.imag, 'imag')

_add_method(Tensor)(T.conjugate, 'conj')
_add_method(Tensor)(T.conjugate, 'conjugate')
_add_method(Tensor)(T.cast, 'cast')
_add_method(Tensor)(T.cast, 'astype')
_add_method(Tensor)(T.squeeze, 'squeeze')
_add_method(Tensor)(T.flatten, 'flatten')
_add_method(Tensor)(T.reshape, 'reshape')
_add_method(Tensor)(T.T, 'T')
_add_method(Tensor)(T.T, 'transpose')
_add_method(Tensor)(T.dot, 'dot')
_add_method(Tensor)(T.repeat, 'repeat')
_add_method(Tensor)(T.expand_dims, 'expand_dims')
_add_method(Tensor)(T.matmul, 'matmul')
_add_method(Tensor)(T.round, 'round')
