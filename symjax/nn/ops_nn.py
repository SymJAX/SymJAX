import inspect
import sys

import jax
import jax.lax as jla
import numpy

from symjax.tensor import jax_wrap

NAMES = [c[0] for c in inspect.getmembers(jax.nn, callable)]
module = sys.modules[__name__]
for name in NAMES:
    if name == "one_hot":
        continue
    module.__dict__.update({name: jax_wrap(jax.nn.__dict__[name])})


def log_1_minus_sigmoid(x):
    return -module.__dict__["softplus"](x)
