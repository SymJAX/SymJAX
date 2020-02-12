import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import numpy
import jax
import jax.lax as jla
from .base import Op, Tuple, jax_wrap
import ast
import inspect
import sys


names = ['det', 'inv', 'norm', 'eigh', 'eig', 'qr', 'svd', 'cholesky']
module = sys.modules[__name__]
for name in names:
    module.__dict__.update(
        {name: jax_wrap(jnpl.__dict__[name])})




