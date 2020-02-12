import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import numpy
import jax
import jax.lax as jla
from .base import Op, Tuple
import ast
import inspect
import sys


names = ['det', 'inv', 'norm']
module = sys.modules[__name__]
for name in names:
    module.__dict__.update(
        {name: type(name, (Op,), {'_fn': staticmethod(jnpl.__dict__[name])})})
    module.__dict__[name].__doc__ = jnpl.__dict__[name].__doc__



def cholesky(a):
    return Tuple(jnpl.cholesky, a)

def eig(a):
    return Tuple(jnpl.eig, a)

def eigh(a):
    return Tuple(jnpl.eigh, a)

def qr(a):
    return Tuple(jnpl.qr, a)

def svd(a, compute_uv=True):
    return Tuple(jnpl.svd, args=[a],
                 kwargs={'compute_uv': compute_uv})



