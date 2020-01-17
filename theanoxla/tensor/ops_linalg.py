import jax.numpy as jnp
import numpy
import jax
import jax.lax as jla
from .base import Op, Tuple, add_fn
from .control_flow import cond
import ast
import inspect


def create_generic_class(func):                                                              
    name = func.split('.')[-1]
    exec('global {}\nclass {}(Op):\n\tpass\nadd_fn({})({})'.format(name, name,               
                                                                   name, func))


names = ['det', 'inv', 'norm']

for name in names:
    create_generic_class('jnp.linalg.' + name)


def cholesky(a):
    return Tuple(jnp.linalg.cholesky, a)

def eig(a):
    return Tuple(jnp.linalg.cholesky, a)

def eigh(a):
    return Tuple(jnp.linalg.cholesky, a)

def qr(a):
    return Tuple(jnp.linalg.cholesky, a)

def svd(a, compute_uv=True):
    return Tuple(jnp.linalg.cholesky, args=[a],
                 kwargs={'compute_uv': compute_uv})



