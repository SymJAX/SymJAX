import jax.numpy as jnp
import jax.lax as jla
from .base import Op, Tensor, theanofn_to_jaxfn
from .base import Tensor
import numpy

# basic arithmetic operators
add = Op(jnp.add, name='add')
sub = Op(jnp.subtract, name='sub')
mul = Op(jnp.multiply, name='mul')
div = Op(jnp.divide, name='div')

# basic comparison operators
eq = Op(jnp.equal, name='equal')
geq = Op(jnp.greater_equal, name='geq')
leq = Op(jnp.less_equal, name='leq')
gr = Op(jnp.greater, name='greater')
le = Op(jnp.less, name='less')
neq = Op(jnp.not_equal, name='different')

# access operator
getitemop = Op(jnp.lax_numpy._rewriting_take)

def _getitem(obj, key):
    # the dtype never changes from accessing
    dtype=obj.dtype
    # first the case where the given key is a list of indices
    if type(key) == list:
        assert numpy.max(key) < obj.shape[0]
        shape = (len(key), ) + obj.shape[1:]
        return getitemop(obj, key, _shape=shape, _dtype=dtype)
    elif type(key) == slice:
        shape = (len(range(*key.indices(obj.shape[0]))),) + obj.shape[1:]
        return getitemop(obj, key, _shape=shape, _dtype=dtype)
    elif numpy.isscalar(key):
        assert key < obj.shape[0]
        shape = obj.shape[1:]
        return getitemop(obj, key, _shape=shape, _dtype=dtype)

    # we now consider the case of having multiple elements
    # first we transform all the elements into the new shape
    new_shape = tuple([len(range(*k.indices(dim)))
                       for k, dim in zip(key, obj.shape[:len(key)])])
    new_shape += obj.shape[len(key):]
    return getitemop(obj, key, _shape=new_shape, _dtype=dtype)


# overloading the getattr method
Tensor.__getitem__ = _getitem
# overloading the basic arithmetic operators
Tensor.__add__ = lambda obj, other: add(obj, other)
Tensor.__radd__ = Tensor.__add__
Tensor.__sub__ = lambda obj, other: sub(obj, other)
Tensor.__rsub__ = lambda obj, other: sub(other, obj)
Tensor.__mul__ = lambda obj, other: mul(obj, other)
Tensor.__rmul__ = Tensor.__mul__
Tensor.__truediv__ = lambda obj, other: div(obj, other)
Tensor.__rtruediv__ = lambda obj, other: div(other, obj)
# overloading comparison operators
Tensor.__eq__ = lambda obj, other: eq(obj, other)
Tensor.__req__ = Tensor.__eq__
Tensor.__lt__ = lambda obj, other: le(obj, other)
Tensor.__rlt__ = Tensor.__gt__
Tensor.__gt__ = lambda obj, other: gr(obj, other)
Tensor.__rgt__ = Tensor.__lt__
Tensor.__ge__ = lambda obj, other: geq(obj, other)
Tensor.__rge__ = Tensor.__le__
Tensor.__le__ = lambda obj, other: leq(obj, other)
Tensor.__rle__ = Tensor.__ge__
Tensor.__ne__ = lambda obj, other: neq(obj, other)
Tensor.__rne__ = Tensor.__ne__



#

zeros = Op(jnp.zeros, name='zeros')
ones = Op(jnp.zeros, name='ones')
full = Op(jla.full, name='full')


# other
cos = Op(jnp.cos, name='cos')
sum = Op(jnp.sum, name='sum')
identity = lambda x:x
matmul = Op(jnp.matmul, name='matmul')
reshape = Op(jnp.reshape, name='reshape')
square =  Op(jnp.square, name='square')
sqrt =  Op(jnp.sqrt, name='sqrt')
arange = Op(jnp.arange, name='range')
pow =  Op(jla.pow, name='pow')
flatten = lambda input: reshape(input, (-1,))
flatten2d = lambda input: reshape(input, (input.shape[0], -1))

_map = Op(jla.map, name='map')
def map(fn, xs):
    newfn = lambda x, _fn=fn: theanofn_to_jaxfn(*x, _fn=_fn)
    return _map(newfn, xs)

_scan = Op(jla.scan, name='scan')
def scan(fn, init, xs):
    newfn = lambda _init, _x, _fn=fn: theanofn_to_jaxfn(_init, *_x, _fn=_fn)
    return _scan(newfn, init, xs)

_cond = Op(jla.cond, name='cond')
def cond(predicate, true_predicate, true_fun, false_predicate, false_fun):
    """ predicate should be a boolean tensor with shape ()
    true_input is the input passed to true_fn that will give the output
    if the predicate evaluates to True, and conversely for False..."""

    # in case the given predicates are not tuples, set them
    if type(true_predicate) != tuple:
        if type(true_predicate) == list:
            true_predicate = tuple(true_predicate)
        else:
            true_predicate = (true_predicate, )
    if type(false_predicate) != tuple:
        if type(false_predicate) == list:
            false_predicate = tuple(false_predicate)
        else:
            false_predicate = (false_predicate, )


    newtruefn = lambda x, _fn=true_fun: theanofn_to_jaxfn(*x, _fn=_fn)
    newfalsefn = lambda x, _fn=false_fun: theanofn_to_jaxfn(*x, _fn=_fn)

    op = _cond(predicate, true_predicate, newtruefn, false_predicate,
               newfalsefn)
    return op


_cast = Op(jla.convert_element_type, 'cast')
def cast(element, dtype):
    return _cast(operand=element, new_dtype=dtype, _shape=element.shape, _dtype=dtype)

# conv
conv_general_dilated_op = Op(jla.conv_general_dilated,
                           name='conv_general_dilated')

def convNd(input, filter, strides=1, padding='VALID', input_format=None,
           filter_format=None, output_format=None, input_dilation=None,
           filter_dilation=None):
    """General n-dimensional convolution operator, with optional dilation.

    Wraps Jax's conv_general_dilated functin, and thus also the XLA's `Conv
    <https://www.tensorflow.org/xla/operation_semantics#conv_convolution>`_
    operator.

    Args:
        input (Tensor): a rank `n+2` dimensional input array.
        filter (Tensor): a rank `n+2` dimensional array of kernel weights.
        strides (int, sequence of int, optional): a (sequence) of `n` integers,
            representing the inter-window strides. If a scalar is given, it is
            used `n` times. Defaults to `1`.
        padding (sequence of couple, `'SAME'`, `'VALID'`, optional): a sequence of
            `n` `(low, high)` integer pairs that give the padding to apply
            before and after each spatial dimension. For  `'VALID'`, those are
            `0`. For `'SAME'`, they are the `input length - filter length + 1`
            for each dim. Defaults to `'Valid'`.
        input_format (`None` or str, optional): a string of same length as the
            number of dimensions in `input` which specify their role
            (see below). Defaults to `'NCW'` for 1d conv, `'NCHW'` for 2d conv,
             and `'NDCHW'` for 3d conv.
        input_dilation (`None`, int or sequence of int, optional): giving the
            dilation factor to apply in each spatial dimension of `input`.
            Inumpy.t dilation is also known as transposed convolution as it allows
            to increase the output spatial dimension by inserting in the input
            any number of `0`s between each spatial value.
        filter_dilation (`None`, int or sequence of int): giving the dilation
            factor to apply in each spatial dimension of `filter`. Filter
            dilation is also known as atrous convolution as it corresponds to
            inserting any number of `0`s in between the filter values, similar
            to performing the non-dilated filter convolution with a subsample
            version of the input across the spatial dimensions.

    Returns:
        Tensor: An array containing the convolution result.

    Format of `input`, `filter` and `output`:
    For example, to indicate dimension numbers consistent with the `conv` function
    with two spatial dimensions, one could use `('NCHW', 'OIHW', 'NCHW')`. As
    another example, to indicate dimension numbers consistent with the TensorFlow
    Conv2D operation, one could use `('NHWC', 'HWIO', 'NHWC')`. When using the
    latter form of convolution dimension specification, window strides are
    associated with spatial dimension character labels according to the order in
    which the labels appear in the `rhs_spec` string, so that `window_strides[0]`
    is matched with the dimension corresponding to the first character
    appearing in rhs_spec that is not `'I'` or `'O'`.
    """
    # setting up the strides
    if numpy.isscalar(strides):
        strides = (strides,) * (input.ndim-2)
    elif len(strides) != (input.ndim - 2):
        msg = 'given strides: {} should match the number'.format(strides) +\
              'of spatial dim. in input: {}'.format(input.ndim-2)
        raise ValueError(msg)

    # setting up the padding
    if type(padding) != str:
        strides = (strides,) * (input.ndim-2)
        if len(padding) != (input.ndim - 2):
            msg = 'given padding: {} should match the '.format(padding) +\
                  'number of spatial dim. in input: {}'.format(input.ndim-2)
            raise ValueError(msg)


    # setting up the filter_format
    if filter_format is None:
        if filter.ndim == 3:
            filter_format = 'OIW'
        elif filter.ndim == 4:
            filter_format = 'OIHW'
        elif filter.ndim == 5:
            filter_format = 'OIDHW'
        else:
            msg = 'filter_format should be given for >5 dimensions.'
            raise ValueError(msg)
    elif len(filter_format) != filter.ndim:
        msg = 'given filter_format: {} should'.format(len(filter_format)) +\
              'match the number of dimension in filter: {}'.format(filter.ndim)
        raise ValueError(msg)

    # setting up the input format
    if input_format is None:
        if len(filter.shape) == 3:
            input_format = 'NCW'
        elif len(filter.shape) == 4:
            input_format = 'NCHW'
        elif len(filter.shape) == 5:
            input_format = 'NCDHW'
        else:
            msg = 'input_format should be given for >5 dimensions.'
            raise ValueError(msg)
    elif len(input_format) != input.ndim:
        msg = 'given input_format: {} should'.format(len(input_format)) +\
              'match the number of dimension in input: {}'.format(input.ndim)
        raise ValueError(msg)


    # setting up the output format
    if output_format is None:
        if len(filter.shape) == 3:
            output_format = 'NCW'
        elif len(filter.shape) == 4:
            output_format = 'NCHW'
        elif len(filter.shape) == 5:
            output_format = 'NCDHW'
        else:
            msg = 'output_format should be given for >5 dimensions.'
            raise ValueError(msg)
    elif len(output_format) != input.ndim:
        msg = 'given output_format: {} should'.format(len(output_format)) +\
              'match the number of dimension in output: {}'.format(input.ndim)
        raise ValueError(msg)


    specs = (input_format, filter_format, output_format)
    output_shape = jla.conv_general_shape_tuple(lhs_shape=input.shape,
                                                rhs_shape=filter.shape,
                                                window_strides=strides,
                                                padding=padding,
                                                dimension_numbers=specs)
    output_dtype = 'float32'
    return conv_general_dilated_op(lhs=input, rhs=filter, window_strides=strides,
                                 padding=padding, lhs_dilation=input_dilation,
                                 rhs_dilation=filter_dilation,
                                 dimension_numbers=specs, precision=None,
                                 _shape=output_shape, _dtype=output_dtype)


#conv_transpose_shape_tuple(lhs_shape, rhs_shape, window_strides, padding,
#                             dimension_numbers)

# pooling
pool_op = Op(jla.reduce_window)

def pool(input, window_shape, reducer='MAX', strides=None, padding='VALID',
          init_val=None, rescalor=None):

    # set up the init_val if not given
    if reducer == 'MAX' and init_val is None:
        init_val = -numpy.inf
    elif (reducer == 'SUM' or reducer=='AVG') and init_val is None:
        init_val = 0.

    # set up rescalor
    if reducer == 'AVG':
        rescalor = numpy.float32(1./numpy.prod(window_shape))
    else:
        rescalor = numpy.float32(1.)

    # set up the reducer
    if reducer == 'MAX':
        reducer = jla.max
    elif reducer == 'SUM' or reducer == 'AVG':
        reducer = jla.add

    # set up the window_shape
    if numpy.isscalar(window_shape):
        window_shape = (window_shape,) * input.ndim
    elif len(window_shape) != input.ndim:
        msg = 'Given window_shape {} not the same length '.format(strides) +\
              'as input shape {}'.format(input.ndim)
        raise ValueError(msg)

    # set up the strides
    if strides is None:
        strides = window_shape
    elif numpy.isscalar(strides):
        strides = (strides,) * len(window_shape)
    elif len(strides) != len(window_shape):
        msg = 'Given strides {} not the same length '.format(strides) +\
              'as window_shape {}'.format(window_shape)
        raise ValueError(msg)

    out_shape = jla.reduce_window_shape_tuple(input.shape, window_shape,
                                              strides, padding)
    out_dtype = input.dtype

    out = pool_op(operand=input*rescalor, init_value=init_val,
                  computation=reducer, window_dimensions=window_shape,
                  window_strides=strides, padding=padding,
                  _shape=out_shape, _dtype=out_dtype)
    return out


