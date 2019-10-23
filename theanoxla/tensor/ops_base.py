import jax.numpy as jnp
import jax.lax as jla
from .base import Op

# access operator
getitemop = Op(jnp.lax_numpy._rewriting_take, name='getitem')


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

# basic comparison operators
eq = Op(jnp.equal, name='equal')
geq = Op(jnp.greater_equal, name='geq')
leq = Op(jnp.less_equal, name='leq')
gr = Op(jnp.greater, name='greater')
le = Op(jnp.less, name='less')
neq = Op(jnp.not_equal, name='different')

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
