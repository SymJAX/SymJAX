import jax.numpy as jnp
import jax.lax as jla

from .base import Op, Tensor, Variable, jax_wrap
from .control_flow import cond
from .ops_math import dynamic_slice_in_dim
import numpy


# conv
conv_general_dilated = jax_wrap(jla.conv_general_dilated)

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
    return conv_general_dilated(lhs=input, rhs=filter, window_strides=strides,
                                 padding=padding, lhs_dilation=input_dilation,
                                 rhs_dilation=filter_dilation,
                                 dimension_numbers=specs, precision=None)



# pooling
reduce_window = jax_wrap(jla.reduce_window)

def poolNd(input, window_shape, reducer='MAX', strides=None, padding='VALID',
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

#    out_shape = jla.reduce_window_shape_tuple(input.shape, window_shape,
#                                              strides, padding)
#    out_dtype = input.dtype

    out = reduce_window(operand=input*rescalor, init_value=init_val,
                  computation=reducer, window_dimensions=window_shape,
                  window_strides=strides, padding=padding)
    return out

def ExponentialMovingAverage(value, alpha, step=None, init=None):
    if step is None:
        _step = Variable(0, trainable=False, name='step')
    else:
        _step = step
    if init is None:
        var = Variable(numpy.zeros(value.shape), trainable=False,
                              name='EMA')
    else:
        var = Variable(init, trainable=False, name='EMA')

    new_value = cond(_step == 0, value, lambda x: x,
                     (var, alpha, value), lambda a, b, c:
                     a * b + (1 - b) * c)
    if step is None:
        updates = {var: new_value, _step: _step + 1}
    else:
        updates = {var: new_value}
    return var, updates, _step

# con

def PiecewiseConstant(init, values, step=None):
    """
    init it the initial value
    values is a dictionnary mapping the knots and the new value
    step count the number of steps
    """
    if step is None:
        step = Variable(0, trainable=False, name='PiecewiseConstant_step')
    keys, values = list(values.keys()), list(values.values())
    keys.insert(0, 0)
    values.insert(0, init)
    keys, values = numpy.array(keys), numpy.array(values)
    assert numpy.prod(keys >= 0)
    arg = numpy.argsort(keys)
    keys, values = keys[arg], values[arg]
    index = (step < keys).argmax()
    v = Variable(values, trainable=False, name='PiecewiseConstant_values')
    return dynamic_slice_in_dim(v, index-1, 1, 0), step

