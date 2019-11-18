import jax.numpy as jnp
import jax.lax as jla
from .base import Tensor, Op

from . import ops_base, ops_math


getitemop = Op(jnp.lax_numpy._rewriting_take, name='getitem')
Tensor.__getitem__ = lambda obj, idx: getitemop(obj, idx)

# overloading the basic arithmetic operators
Tensor.__add__ = lambda obj, other: ops_base.add(obj, other)
Tensor.__radd__ = Tensor.__add__
Tensor.__sub__ = lambda obj, other: ops_base.sub(obj, other)
Tensor.__rsub__ = lambda obj, other: ops_base.sub(other, obj)
Tensor.__mul__ = lambda obj, other: ops_base.mul(obj, other)
Tensor.__rmul__ = Tensor.__mul__
Tensor.__truediv__ = lambda obj, other: ops_base.div(obj, other)
Tensor.__rtruediv__ = lambda obj, other: ops_base.div(other, obj)
Tensor.__neg__ = lambda obj: ops_base.sub(0, obj)
Tensor.__pow__ = lambda obj, other: ops_math.pow(obj, other)
# overloading comparison operators
#Tensor.__eq__ = lambda obj, other: ops_base.eq(obj, other)
#Tensor.__req__ = Tensor.__eq__
Tensor.__lt__ = lambda obj, other: ops_base.le(obj, other)
Tensor.__rlt__ = lambda obj, other: Tensor.__gt__(obj, other)
Tensor.__gt__ = lambda obj, other: ops_base.gr(obj, other)
Tensor.__rgt__ = Tensor.__lt__
Tensor.__ge__ = lambda obj, other: ops_base.geq(obj, other)
Tensor.__rge__ = lambda obj, other: Tensor.__le__(obj, other)
Tensor.__le__ = lambda obj, other: ops_base.leq(obj, other)
Tensor.__rle__ = lambda obj, other: Tensor.__ge__(obj, other)
#Tensor.__ne__ = lambda obj, other: ops_base.neq(obj, other)
#Tensor.__rne__ = lambda obj, other: Tensor.__ne__(obj, other)

# additional operators
Tensor.sum = lambda obj, *args, **kwargs: ops_math.sum(obj, *args, **kwargs)
Tensor.mean = lambda obj, *args, **kwargs: ops_math.mean(obj, *args, **kwargs)
Tensor.max = lambda obj, *args, **kwargs: ops_math.max(obj, *args, **kwargs)
Tensor.argmax = lambda obj, *args, **kwargs: ops_math.argmax(obj, *args, **kwargs)
Tensor.std = lambda obj, *args, **kwargs: ops_math.std(obj, *args, **kwargs)
Tensor.var = lambda obj, *args, **kwargs: ops_math.var(obj, *args, **kwargs)


Tensor.astype = lambda obj, dtype: ops_base.cast(obj, dtype)
Tensor.prod = lambda obj, *args, **kwargs: ops_math.prod(obj, *args, **kwargs)
Tensor.squeeze = lambda obj: ops_math.squeeze(obj)
Tensor.flatten = lambda obj: ops_math.flatten(obj)
Tensor.reshape = lambda obj, new_shape: ops_math.reshape(obj, new_shape)
Tensor.T = lambda obj: ops_math.transpose(obj)
Tensor.dot = lambda obj, other: ops_math.dot(obj, other)
Tensor.transpose = lambda  obj, *args, **kwargs: ops_math.transpose(obj, *args, **kwargs)
