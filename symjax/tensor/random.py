import jax.random as jnp
from .base import jax_wrap
import sys



_RANDOM_FUNCTIONS = [jnp.bernoulli, jnp.beta, jnp.cauchy,
                     jnp.dirichlet, jnp.gamma, jnp.gumbel,
                     jnp.laplace, jnp.logit, jnp.categorical,
                     jnp.multivariate_normal, jnp.normal,
                     jnp.pareto, jnp.randint, jnp.shuffle,
                     jnp.threefry_2x32, jnp.truncated_normal,
                     jnp.uniform]




module = sys.modules[__name__]

for name in _RANDOM_FUNCTIONS:
    module.__dict__.update({name.__name__: jax_wrap(name)})

randn = jax_wrap(jnp.normal)

