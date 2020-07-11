import inspect
import sys

import jax.numpy.linalg as jnpl
import jax.scipy.linalg as jspl

from .base import jax_wrap

from_scipy = [
    "cholesky",
    "block_diag",
    "cho_solve",
    "eigh",
    "expm",
    #   "expm_frechet",
    "inv",
    "lu",
    "lu_factor",
    "lu_solve",
    "solve_triangular",
    "tril",
    "triu",
]

NAMES = [c[0] for c in inspect.getmembers(jnpl, inspect.isfunction)]
module = sys.modules[__name__]
for name in NAMES:
    if name not in from_scipy:
        module.__dict__.update({name: jax_wrap(jnpl.__dict__[name])})

for name in from_scipy:
    module.__dict__.update({name: jax_wrap(jspl.__dict__[name])})
