import jax.numpy as jnp
import jax.lax as jla
from .base import Op, Tensor, Slice
from .ops_math import concatenate, reshape

# access operator
getitemop = Op(jnp.lax_numpy._rewriting_take, name='getitem')
# overloading the getattr method
def turn2Slice(item):
    if type(item) == tuple:
        return tuple([turn2Slice(i) for i in item])
    elif type(item) == slice:
        print('TURNING')
        return Slice(item)
    else:
        return item

def getitem(obj, *idx):
    print('idx before', idx)
    idx = turn2Slice(idx)
    print('idx after', idx)
    return getitemop(obj, *idx)


Tensor.__getitem__ = getitem#lambda obj, idx: getitemop(obj, idx)


gather = Op(jla.gather, name='gather')
take = Op(jnp.take, name='take')

slice = Op(jla.slice, name='slice')
dynamic_slice = Op(jla.dynamic_slice, name='dynamic_slice')
slice_in_dim = Op(jla.slice_in_dim, name='slice_in_dim')
dynamic_slice_in_dim = Op(jla.dynamic_slice_in_dim, name='dynamic_slice_in_dim')



# basic arithmetic operators
add = Op(jnp.add, name='add')
sub = Op(jnp.subtract, name='sub')
mul = Op(jnp.multiply, name='mul')
div = Op(jnp.divide, name='div')
# overloading the basic arithmetic operators
Tensor.__add__ = lambda obj, other: add(obj, other)
Tensor.__radd__ = Tensor.__add__
Tensor.__sub__ = lambda obj, other: sub(obj, other)
Tensor.__rsub__ = lambda obj, other: sub(other, obj)
Tensor.__mul__ = lambda obj, other: mul(obj, other)
Tensor.__rmul__ = Tensor.__mul__
Tensor.__truediv__ = lambda obj, other: div(obj, other)
Tensor.__rtruediv__ = lambda obj, other: div(other, obj)

# basic comparison operators
eq = Op(jnp.equal, name='equal')
geq = Op(jnp.greater_equal, name='geq')
leq = Op(jnp.less_equal, name='leq')
gr = Op(jnp.greater, name='greater')
le = Op(jnp.less, name='less')
neq = Op(jnp.not_equal, name='different')
# overloading comparison operators
Tensor.__eq__ = lambda obj, other: eq(obj, other)
Tensor.__req__ = Tensor.__eq__
Tensor.__lt__ = lambda obj, other: le(obj, other)
Tensor.__rlt__ = Tensor.__gt__
Tensor.__gt__ = lambda obj, other: gr(obj, other)
Tensor.__rgt__ = Tensor.__lt__
Tensor.__ge__ = lambda obj, other: geq(obj, other)
Tensor.__rge__ = Tensor.__le__
Tensor.__le__ = lambda obj, other: leq(obj, other)
Tensor.__rle__ = Tensor.__ge__
Tensor.__ne__ = lambda obj, other: neq(obj, other)
Tensor.__rne__ = Tensor.__ne__



# tensor creation
eye = Op(jnp.eye, name='eye')

zeros = Op(jnp.zeros, name='zeros')
zeros_like = Op(jnp.zeros_like, name='zeros_like')

empty = Op(jnp.empty, name='empty')
empty_like = Op(jnp.empty_like, name='empty_like')

ones = Op(jnp.ones, name='ones')
ones_like = Op(jnp.ones_like, name='ones_like')

full = Op(jla.full, name='full')
full_like  = Op(jla.full_like, name='full_like')

# cast operator
cast = Op(jla.convert_element_type, 'cast')
