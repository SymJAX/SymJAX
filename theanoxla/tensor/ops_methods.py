import jax.numpy as jnp
import jax.lax as jla
from .base import Tensor

import .ops_base
import .ops_math



Tensor.__getitem__ = getitemop

# overloading the basic arithmetic operators
Tensor.__add__ = ops_base.add
Tensor.__radd__ = Tensor.__add__
Tensor.__sub__ = ops_base.sub
Tensor.__rsub__ = lambda obj, other: ops_base.sub(other, obj)
Tensor.__mul__ = ops_base.mul
Tensor.__rmul__ = Tensor.__mul__
Tensor.__truediv__ = ops_base.div
Tensor.__rtruediv__ = lambda obj, other: ops_base.div(other, obj)

# overloading comparison operators
Tensor.__eq__ = ops_base.eq
Tensor.__req__ = Tensor.__eq__
Tensor.__lt__ = ops_base.le
Tensor.__rlt__ = Tensor.__gt__
Tensor.__gt__ = ops_base.gr
Tensor.__rgt__ = Tensor.__lt__
Tensor.__ge__ = ops_base.geq
Tensor.__rge__ = Tensor.__le__
Tensor.__le__ = ops_base.leq
Tensor.__rle__ = Tensor.__ge__
Tensor.__ne__ = ops_base.neq
Tensor.__rne__ = Tensor.__ne__

# additional operators
Tensor.sum = ops_math.sum
Tensor.prod = ops_math.prod


