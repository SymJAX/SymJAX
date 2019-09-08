import jax.numpy as jnp
import jax.lax as jla
from .base import Op
from .base import Tensor

# basic arithmetic operators
add = Op(jnp.add, name='add')
sub = Op(jnp.subtract, name='sub')
mul = Op(jnp.multiply, name='mul')
div = Op(jnp.divide, name='div')

# basic comparison operators
equal = Op(jnp.equal, name='equal')
geq = Op(jnp.greater_equal, name='geq')
leq = Op(jnp.less_equal, name='leq')
greater = Op(jnp.greater, name='greater')
less = Op(jnp.less, name='less')
neq = Op(jnp.not_equal, name='different')

# overloading the basic arithmetic operators
Tensor.__add__ = lambda obj, other: add(obj, other)
Tensor.__radd__ = Tensor.__add__
Tensor.__sub__ = lambda obj, other: sub(obj, other)
Tensor.__rsub__ = Tensor.__sub__
Tensor.__mul__ = lambda obj, other: mul(obj, other)
Tensor.__rmul__ = Tensor.__mul__
Tensor.__div__ = lambda obj, other: div(obj, other)
Tensor.__rdiv__ = Tensor.__div__
# overloading comparison operators
Tensor.__eq__ = lambda obj, other: equal(obj, other)
Tensor.__req__ = Tensor.__eq__
Tensor.__lt__ = lambda obj, other: less(obj, other)
Tensor.__rlt__ = Tensor.__lt__
Tensor.__gt__ = lambda obj, other: greater(obj, other)
Tensor.__rgt__ = Tensor.__gt__
Tensor.__ge__ = lambda obj, other: geq(obj, other)
Tensor.__rge__ = Tensor.__ge__
Tensor.__le__ = lambda obj, other: leq(obj, other)
Tensor.__rle__ = Tensor.__le__
Tensor.__ne__ = lambda obj, other: neq(obj, other)
Tensor.__rne__ = Tensor.__ne__



#

# other
cos = Op(jnp.cos, name='cos')
sum = Op(jnp.sum, name='sum')


_cast = Op(jla.convert_element_type, 'cast')
def cast(element, dtype):
    return _cast(operand=element, new_dtype=dtype, _shape=element.shape, _dtype=dtype)

