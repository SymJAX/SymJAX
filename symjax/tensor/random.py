import sys

import jax.random as jnp

from .base import jax_wrap

_seed = 0

_RANDOM_FUNCTIONS = [
    jnp.bernoulli,
    jnp.beta,
    jnp.categorical,
    jnp.cauchy,
    jnp.dirichlet,
    jnp.exponential,
    jnp.gamma,
    jnp.gumbel,
    jnp.laplace,
    jnp.logistic,
    jnp.multivariate_normal,
    jnp.normal,
    jnp.pareto,
    jnp.permutation,
    jnp.poisson,
    jnp.randint,
    jnp.shuffle,
    jnp.t,
    jnp.threefry_2x32,
    jnp.truncated_normal,
    jnp.uniform,
]

module = sys.modules[__name__]

for name in _RANDOM_FUNCTIONS:
    module.__dict__.update({name.__name__: jax_wrap(name)})

randn = module.__dict__["normal"]
rand = module.__dict__["uniform"]
