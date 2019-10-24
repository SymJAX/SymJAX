import jax.numpy as jnp
import jax.lax as jla
from .base import Op, List
from .control_flow import cond


#
def patchesop(signal, h, hop=1, data_format='NCW'):
    if data_format == 'NCW':
        N = (signal.shape[2] - h ) // hop +1
        indices = jnp.repeat(jnp.reshape(jnp.arange(h), (1, h)), N, 0)
        indices = indices + jnp.reshape(jnp.arange(0, N), (N, 1)) * hop
        indices = jnp.reshape(indices, [1, 1, N * h])
        patches = jnp.take_along_axis(signal, indices, 2)
        return jnp.reshape(patches, signal.shape[:2] + (N, h))
    else:
        error

extract_signal_patches = Op(patchesop, name='wvd')

stop_gradient = Op(jla.stop_gradient, name='stop_gradient')

# other
cos = Op(jnp.cos, name='cos')
sum = Op(jnp.sum, name='sum')
mean = Op(jnp.mean, name='mean')
max = Op(jnp.max, name='max')
argmax = Op(jnp.argmax, name='argmax')


prod = Op(jnp.prod, name='prod')
linspace = Op(jnp.linspace, name='linspace')
exp = Op(jnp.exp, name='exp')
log = Op(jnp.log, name='log')
abs = Op(jnp.abs, name='abs')
dot = Op(jnp.dot, name='dot')
equal = Op(jnp.equal, name='equal')
expand_dims = Op(jnp.expand_dims, name='expand_dims')

def meshgrid(*args):
    # compute the shape and type of the gradients to create the List
    shapes = tuple([vector.shape[0] for vector in args])
    dtypes = 'float32'
    return List(jnp.meshgrid, [shapes]*len(args), [dtypes]*len(args),
                       args=args)



identity = lambda x:x
matmul = Op(jnp.matmul, name='matmul')
reshape = Op(jnp.reshape, name='reshape')
concatenate = Op(jnp.concatenate, name='concatenate')
stack =  Op(jnp.stack, name='stack')
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
transpose = Op(jnp.transpose, name='transpose')
fliplr = Op(jnp.fliplr, name='fliplr')
flipud = Op(jnp.flipud, name='flipud')
flip = Op(jnp.flip, name='flip')

fft = Op(jla.fft, name='fft')
roll = Op(jnp.roll, name='roll')
pow =  Op(jla.pow, name='pow')
flatten = lambda input: reshape(input, (-1,))
flatten2d = lambda input: reshape(input, (input.shape[0], -1))




