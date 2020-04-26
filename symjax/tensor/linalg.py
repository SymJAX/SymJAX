import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import numpy
import jax
import jax.lax as jla
from .base import Op, Tuple, jax_wrap
import ast
import inspect
import sys


NAMES = [c[0] for c in inspect.getmembers(jnpl, inspect.isfunction)]
module = sys.modules[__name__]
for name in NAMES:
    module.__dict__.update(
        {name: jax_wrap(jnpl.__dict__[name])})




