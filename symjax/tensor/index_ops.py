import jax
import sys
from .base import jax_wrap
module = sys.modules[__name__]

index = jax.ops.index
for name in ['index_update', 'index_min', 'index_add', 'index_max']:
    setattr(module, name, jax_wrap(jax.ops.__dict__[name]))

