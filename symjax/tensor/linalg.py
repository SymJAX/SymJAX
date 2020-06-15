import inspect
import sys

import jax.numpy.linalg as jnpl

from .base import jax_wrap

NAMES = [c[0] for c in inspect.getmembers(jnpl, inspect.isfunction)]
module = sys.modules[__name__]
for name in NAMES:
    module.__dict__.update({name: jax_wrap(jnpl.__dict__[name])})
