import jax.numpy as jnp
from .base import Op
from .base import Tensor

cos = Op(jnp.cos, name='cos')
add = Op(jnp.add, name='add')
sum = Op(jnp.sum, name='sum')
sub = Op(jnp.subtract, name='sub')
mul = Op(jnp.multiply, name='mul')


def __add__(self, other):
    return add(self, other)
def __radd__(self, other):
    return add(self.other)
def __sub__(self, other):
    return sub(self, other)
def __rsub__(self, other):
    return sub(self.other)
def __mul__(self, other):
    return mul(self, other)
def __rmul__(self, other):
    return mul(self.other)

Tensor.__add__ = __add__
Tensor.__radd__ = __radd__
Tensor.__sub__ = __sub__
Tensor.__rsub__ = __rsub__
Tensor.__mul__ = __mul__
Tensor.__rmul__ = __rmul__

