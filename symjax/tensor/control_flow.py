import jax.lax as jla
from .base import jax_wrap

cond = jax_wrap(jla.cond)
fori_loop = jax_wrap(jla.fori_loop)
while_loop = jax_wrap(jla.while_loop)
scan = jax_wrap(jla.scan)
map = jax_wrap(jla.map)

