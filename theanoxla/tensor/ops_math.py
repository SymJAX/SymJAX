import jax.numpy as jnp
import jax.lax as jla
from .base import Op
from .control_flow import cond

# other
cos = Op(jnp.cos, name='cos')
sum = Op(jnp.sum, name='sum')
identity = lambda x:x
matmul = Op(jnp.matmul, name='matmul')
reshape = Op(jnp.reshape, name='reshape')
concatenate = Op(jnp.concatenate, name='concatenate')
square =  Op(jnp.square, name='square')
sqrt = Op(jnp.sqrt, name='sqrt')
real = Op(jnp.real, name='real')

def arangeop(start, stop=None, step=None, dtype='int32'):
    print(start, stop, step, dtype)
    if stop is None and step is None:
        return jla.iota(dtype, start)
    else:
        S = cond( start > stop, [start, stop], lambda i, j:jla.iota(dtype, i-j)[::step]+j,
                 [start, stop], lambda i, j:jla.iota(dtype, j-i)[::step]+i)
        return S
arange = Op(jnp.arange, name='arange')
range = arange

fliplr = Op(jnp.fliplr, name='fliplr')
fft = Op(jla.fft, name='fft')
roll = Op(jnp.roll, name='roll')
pow =  Op(jla.pow, name='pow')
flatten = lambda input: reshape(input, (-1,))
flatten2d = lambda input: reshape(input, (input.shape[0], -1))




