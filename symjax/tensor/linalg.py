import inspect
import sys

import jax.numpy.linalg as jnpl
import jax.scipy.linalg as jspl

from .base import jax_wrap
from .normalization import normalize

from . import random

from_scipy = [
    "cholesky",
    "block_diag",
    "cho_solve",
    "eigh",
    "expm",
    # "expm_frechet",
    "inv",
    "lu",
    "lu_factor",
    "lu_solve",
    "solve_triangular",
    "tril",
    "triu",
]

NAMES = [c[0] for c in inspect.getmembers(jnpl, inspect.isfunction)] + [
    "pinv",
    "slogdet",
]

module = sys.modules[__name__]
for name in NAMES:
    if name not in from_scipy:
        module.__dict__.update({name: jax_wrap(jnpl.__dict__[name])})

for name in from_scipy:
    module.__dict__.update({name: jax_wrap(jspl.__dict__[name])})


def singular_vectors_power_iteration(weight, axis=0, n_iters=1):

    # This power iteration produces approximations of `u` and `v`.

    u = normalize(random.randn(weight.shape[0]), dim=0)
    v = normalize(random.randn(weight.shape[1]), dim=0)

    for _ in range(n_iters):

        v = normalize(weight.t().dot(u), dim=0)
        u = normalize(weight.dot(v), dim=0)

    return u, v


def eigenvector_power_iteration(weight, axis=0, n_iters=1):

    # This power iteration produces approximations of `u`.

    u = normalize(random.randn(weight.shape[0]), dim=0)

    for _ in range(n_iters):

        u = normalize(weight.t().dot(u), dim=0)

    return u
