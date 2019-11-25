import jax.numpy as jnp
import numpy
import jax.lax as jla
from .base import Op, List
from .control_flow import cond
from .ops_activations import relu

#


def hat_1D(x, t_left, t_center, t_right):
    """
    Hat function, continuous piecewise linear, such that::
        f(x) = \begin{cases}
                    0 \iff x \not \in (t_left,t_right)\\
                    1 \iff x = t_center\\
                    \frac{x - t_left}{t_center - t_left} \iff x \in (t_left, t]\\
                    \frac{x - t_center}{t_center - t_right} \iff x \in (t_left, t]
                \end{cases}
    Parameters
    ----------

    x :: array-like
        the sampled input space
    t_left :: scalar
        the position of the left knot
   t_center :: scalar
        the position of the center knot
    t_right :: scalar
        the position of the right knot

    Returns
    -------
    output :: array
        same shape as x with applied hat function
    """

#    if t_left > t_center or t_right < t_center:
#        print('wrong order for the knots')
    slope_left = 1 / (t_center - t_left)
    slope_right = 1 / (t_right - t_center)
    output = (relu(x - t_left)) * slope_left\
            - relu(x - t_center) * (slope_left + slope_right)\
            + relu(x - t_right) * slope_right
    return output


def patchesop(signal, window_length, hop=1, data_format='NCW'):
    assert not hasattr(window_length, '__len__')
    if data_format == 'NCW':
        N = (signal.shape[2] - window_length) // hop +1
        indices = jnp.arange(window_length) +\
                     jnp.expand_dims(jnp.arange(N) * hop, 1)
        indices = jnp.reshape(indices, [1, 1, N * window_length])
        patches = jnp.take_along_axis(signal, indices, 2)
        return jnp.reshape(patches, signal.shape[:2] + (N, window_length))
    else:
        error

def patchesop2(image, window_shape, hop=1, data_format='NCHW', mode='valid'):
    if mode == 'same':
        p1 = (window_shape[0] -1)
        p2 = (window_shape[1] -1)
        image = jnp.pad(image, [(0, 0), (0, 0), (p1 // 2, p1 - p1 // 2),
                                (p2 // 2, p2 - p2 // 2)])
    if not hasattr(hop, '__len__'):
        hop = (hop, hop)
    if data_format == 'NCHW':

        # compute the number of windows in both dimensions
        N = ((image.shape[2] - window_shape[0]) // hop[0] +1,
             (image.shape[3] - window_shape[1]) // hop[1] +1)

        # compute the base indices of a 2d patch
        patch = jnp.arange(numpy.prod(window_shape)).reshape(window_shape)
        offset = jnp.expand_dims(jnp.arange(window_shape[0]), 1)
        patch_indices = patch + offset * (image.shape[3] - window_shape[1])

        # create all the shifted versions of it
        ver_shifts = jnp.reshape(jnp.arange(N[0]) * hop[0] * image.shape[3],
                                 (-1, 1, 1, 1))
        hor_shifts = jnp.reshape(jnp.arange(N[1]) * hop[1], (-1, 1, 1))
        all_cols = patch_indices + jnp.reshape(jnp.arange(N[1]) * hop[1],
                                               (-1, 1, 1))
        indices = patch_indices + ver_shifts + hor_shifts

        # now extract
        flat_indices = jnp.reshape(indices, [1, 1, -1])
        flat_image = jnp.reshape(image, (image.shape[0], image.shape[1], -1))
        patches = jnp.take_along_axis(flat_image, flat_indices, 2)
        return jnp.reshape(patches, image.shape[:2] + N + tuple(window_shape))
    else:
        error



extract_signal_patches = Op(patchesop, name='extract_signal_patches')
extract_image_patches = Op(patchesop2, name='extract_signal_patches')


stop_gradient = Op(jla.stop_gradient, name='stop_gradient')
complex = Op(jla.complex, name='complex')
pad = Op(jnp.pad, name='pad')
add = Op(jnp.add, name='add')
def _add_n(args):
    start = args[0]
    for arg in args:
        start = jnp.add(start, arg)
    return start
add_n = Op(_add_n, name='add_n')
# other
cos = Op(jnp.cos, name='cos')
sin = Op(jnp.sin, name='sin')
conj = Op(jla.conj, name='conj')
sum = Op(jnp.sum, name='sum')
mean = Op(jnp.mean, name='mean')
max = Op(jnp.max, name='max')
argmax = Op(jnp.argmax, name='argmax')
var = Op(jnp.var, name='var')
std = Op(jnp.std, name='std')


def _to_one_hot(vector, n_classes, dtype='float32'):
    return jla.convert_element_type(jnp.equal(jnp.expand_dims(vector,1),
                                              jnp.arange(n_classes)), dtype)

to_one_hot= Op(_to_one_hot, name='to_one_hot')

repeat = Op(jnp.repeat, name='repeat')
prod = Op(jnp.prod, name='prod')
linspace = Op(jnp.linspace, name='linspace')
exp = Op(jnp.exp, name='exp')
log = Op(jnp.log, name='log')
log10 = Op(jnp.log10, name='log10')
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
squeeze = Op(jnp.squeeze, name='squeeze')


def logsumexp(x, axis):
    x_max = stop_gradient(x.max(axis, keepdims=True))
    return log(exp(x - x_max).sum(axis)) + squeeze(x_max)
