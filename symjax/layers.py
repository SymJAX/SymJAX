from .tensor import ops_methods
from . import tensor as T
from . import initializers
import numpy
import inspect


def _is_shape(x):
    """utility function that checks if the input is a shape or not"""
    if not hasattr(x, '__len__'):
        return False
    if isinstance(x[0], int):
        return True
    elif isinstance(x[0], float):
        raise RuntimeError("invalid float value in shape")


def forward(input, layers):
    """ perform a forward path in the layers given an input

    utility function

    Parameters
    ----------

    input: Tensor
        the tensor to be forwarded across the given layers

    layers: list of layers
        the layers to forward the input in a consecutive manner
    """
    outputs = [layers[0].forward(input)]
    for layer in layers[1:]:
        outputs.append(layer.forward(outputs[-1]))
    return outputs


class Layer(T.Tensor):

    def __init__(self, output):
        super().__init__(output.shape, output.dtype, output.roots, copyof=output)

    def variables(self, trainable=True):
        if not hasattr(self, '_variables'):
            return []
        if trainable is not None:
            return [v for v in self._variables if v.trainable == trainable]
        else:
            return self._variables

    def init_input(self, input_or_shape):
        if _is_shape(input_or_shape):
            self.input = T.Placeholder(input_or_shape, 'float32')
        else:
            self.input = input_or_shape

    @property
    def updates(self):
        if not hasattr(self, '_updates'):
            self._updates = {}
        return self._updates

    def add_update(self, update):
        self.updates.update(update)

    def add_variable(self, variable):
        if not hasattr(self, '_variables'):
            self._variables = list()
        self._variables.append(variable)

    def forward(self):
        pass


class Identity(Layer):

    def __init__(self, input_or_shape):

        self.init_input(input_or_shape)
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return input


class Reshape(Layer):

    def __init__(self, input_or_shape, shape):

        self.init_input(input_or_shape)
        self.new_shape = shape
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return T.reshape(input, self.new_shape)


class Upsample(Layer):

    def __init__(self, input_or_shape, factors):

        self.init_input(input_or_shape)
        self.factors = factors
        super().__init__(self.forward(self.input))

    def forward(self, input):
        upsample = T.upsample(input, self.factors)
        return upsample


class Activation(Layer):

    def __init__(self, input_or_shape, activation):

        self.init_input(input_or_shape)
        self.activation = activation
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return self.activation(input)


class Dense(Layer):

    def __init__(self, input_or_shape, units, W=initializers.he,
                 b=numpy.zeros, W_dtype='float32', b_dtype='float32',
                 trainable_W=True, trainable_b=True):

        self.init_input(input_or_shape)

        w_shape = (numpy.prod(self.input.shape[1:]), units)
        self.W = T.Variable(W, shape=w_shape, dtype=W_dtype,
                            trainable=trainable_W)
        self.add_variable(self.W)

        self.b = T.Variable(b, shape=(w_shape[1],), dtype=b_dtype,
                            trainable=trainable_b)
        self.add_variable(self.b)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        if numpy.prod(input.shape[1:]) != self.W.shape[0]:
            raise RuntimeError(
                'input to Dense layer {} has different dim'.format(self))
        return T.dot(T.flatten2d(input), self.W) + self.b


class Conv1D(Layer):
    """ for standard conv1d the W_shape should be
            (#out_filters, #in_channels, time_bins)
        and for the bias it should be (#out_filters, 1) or to not
        share the bias across time put
        (#out_filters, time_bins) or to share all bias put (1,) be
        careful to have the correct
        time for broadcasting if needed
    """
    def __init__(self, input_or_shape, W_shape, b_shape=None, strides=1,
                 pad='VALID', W=initializers.he, b=numpy.zeros,
                 W_dtype='float32', b_dtype='float32',
                 trainable_W=True, trainable_b=True,
                 input_dilations=None, filter_dilations=None):

        self.init_input(input_or_shape)
        self.input_dilation = input_dilations
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.pad = pad

        if not trainable_W:
            self.W = W
        else:
            self.W = T.Variable(W, shape=W_shape, dtype=W_dtype,
                                trainable=trainable_W)
            self.add_variable(self.W)

        if b_shape is None:
            b_shape = (W_shape[0], 1)
        self.b = T.Variable(b, shape=b_shape, dtype=b_dtype,
                            trainable=trainable_b)
        self.add_variable(self.b)

        self.strides = strides

        super().__init__(self.forward(self.input))

    def forward(self, input):
        conv = T.convNd(input, self.W, strides=self.strides, padding=self.pad,
                        input_dilation=self.input_dilation,
                        filter_dilation=self.filter_dilation)
        return conv + self.b


class Conv2D(Layer):
    def __init__(self, input_or_shape, W_shape, b_shape=None, pad='VALID',
                 strides=1, W=initializers.he, b=numpy.zeros,
                 W_dtype='float32', b_dtype='float32',
                 trainable_W=True, trainable_b=True,
                 input_dilations=None, filter_dilations=None):

        self.init_input(input_or_shape)
        self.input_dilation = input_dilations
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.pad = pad

        self.W = T.Variable(W, shape=W_shape, dtype=W_dtype,
                            trainable=trainable_W)
        self.add_variable(self.W)

        if b_shape is None:
            b_shape = (W_shape[0], 1, 1)
        self.b = T.Variable(b, shape=b_shape, dtype=b_dtype,
                            trainable=trainable_b)
        self.add_variable(self.b)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        conv = T.convNd(input, self.W, strides=self.strides, padding=self.pad,
                        input_dilation=self.input_dilation,
                        filter_dilation=self.filter_dilation)

        return conv + self.b


class Pool2D(Layer):
    def __init__(self, input_or_shape, pool_shape, pool_type='MAX',
                 strides=None):

        self.init_input(input_or_shape)
        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape[0], pool_shape[1])
        if strides is None:
            self.strides = self.pool_shape
        else:
            if hasattr(strides, __len__):
                self.strides = (1, 1, strides[0], strides[1])
            else:
                self.strides = (1, 1, strides, strides)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        return T.poolNd(input, self.pool_shape, strides=self.strides,
                        reducer=self.pool_type)


class Dropout(Layer):

    """binary mask onto the input

    Parameters
    ----------

    input_or_shape: shape or Tensor
        the layer input or shape

    p: float (0<=p<=1)
        the probability to drop the value

    deterministic: bool or Tensor
        the state of the layer

    seed: seed
        the RNG seed

    Returns
    -------

    output: the layer output

    """

    def __init__(self, input_or_shape, p, deterministic, seed=None):

        self.init_input(input_or_shape)
        self.deterministic = deterministic
        self.p = p
        self.mask = T.random.bernoulli(shape=self.input.shape, p=p, seed=seed)
        super().__init__(self.forward(self.input))

    def forward(self, input, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic
        dirac = T.cast(deterministic, 'float32')
        return input * self.mask * (1 - dirac) + input * dirac


class RandomFlip(Layer):

    """
    random axis flip on the input

    Random layer that will randomly flip the axis of the input.
    Note that all the involved
    operations are GPU compatible and allow for backpropagation

    Parameters
    ----------

    input_or_shape: shape or Tensor
        the input of the layer or the shape of the layer input

    crop_shape: shape
        the shape of the cropped part of the input. It must have the same
        length as the input shape  minus one for the first dimension

    deterministic: bool or Tensor
        if the layer is in deterministic mode or not

    padding: shape
        the amount of padding to apply on each dimension (except the first
        one) each dimension should have a couple for the before and
        after padding parts

    seed: seed (optional)
        to control reproducibility

    Returns
    -------

    output: the output tensor which containts the internal variables
    """
    
    def __init__(self, input_or_shape, p, axis, deterministic, seed=None):

        self.init_input(input_or_shape)
        self.deterministic = deterministic
        self.p = p
        self.axis = axis
        self.flip = T.random.bernoulli(shape=(input.shape[0]), p=p, seed=seed)
        super().__init__(self.forward(self.input))

    def forward(self, input, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic

        dirac = T.cast(deterministic, 'float32')

        flipped_input = self.flip * T.flip(input, self.axis)\
            + (1 - self.flip) * input

        return input * dirac + flipped_input * (1 - dirac)


class RandomCrop(Layer):

    """
    random crop selection form the input

    Random layer that will select a window of the input based on the given
    parameters, with the possibility to first apply a padding. This layer is
    commonly used as a data augmentation technique and positioned at the
    beginning of the deep network topology. Note that all the involved
    operations are GPU compatible and allow for backpropagation

    Parameters
    ----------

    input_or_shape: shape or Tensor
        the input of the layer or the shape of the layer input

    crop_shape: shape
        the shape of the cropped part of the input. It must have the same
        length as the input shape  minus one for the first dimension

    deterministic: bool or Tensor
        if the layer is in deterministic mode or not

    padding: shape
        the amount of padding to apply on each dimension (except the first
        one) each dimension should have a couple for the before and
        after padding parts

    seed: seed (optional)
        to control reproducibility

    Returns
    -------

    output: the output tensor which containts the internal variables

    """

    def __init__(self, input_or_shape, crop_shape, deterministic, padding=0,                 seed=None):

        self.init_input(input_or_shape)
        self.crop_shape = crop_shape
        # if given only a scalar
        if not hasattr(padding, '__len__'):
            self.pad_shape = [(padding, padding)] * (self.input.shape - 1)
        # else
        else:
            self.pad_shape = [(pad, pad) if not hasattr(pad, '__len__')
                              else pad for pad in padding]

        assert len(self.pad_shape) == len(self.crop_shape)
        assert len(self.pad_shape) == (len(self.input.shape) - 1)

        self.deterministic = deterministic

        self.start_indices = list()
        self.fixed_indices = list()
        for i, (pad, dim, crop) in enumerate(
                zip(self.pad_shape, self.input.shape[1:], self.crop_shape)):
            maxval = pad[0] + pad[1] + dim - crop
            assert maxval >= 0
            self.start_indices.append(
                T.random.randint(minval=0, maxval=maxval,
                    shape=(self.input.shape[0],1),
                    dtype='int32',
                    seed=seed + i if seed is not None else seed))

            self.fixed_indices.append(T.ones((self.input.shape[0],1), 'int32') * (maxval//2))
        self.start_indices = T.concatenate(self.start_indices, 1)
        self.fixed_indices = T.concatenate(self.fixed_indices, 1)

        super().__init__(self.forward(self.input))

    def forward(self, input, deterministic=None):

        if deterministic is None:
            deterministic = self.deterministic
        dirac = T.cast(deterministic, 'float32')

        # pad the input
        pinput = T.pad(input, [(0,0)] + self.pad_shape)

        routput = T.stack(
            [
                T.dynamic_slice(
                    pinput[n],
                    self.start_indices[n],
                    self.crop_shape) for n in range(
                    self.input.shape[0])],
            0)
        doutput = T.stack(
            [
                T.dynamic_slice(
                    pinput[n],
                    self.fixed_indices[n],
                    self.crop_shape) for n in range(
                    self.input.shape[0])],
            0)


        return doutput * dirac +  (1 - dirac) * routput


class BatchNormalization(Layer):
    """
    batch-normalization layer


    Parameters:
    -----------

    input_or_shape: shape or Tensor
        the layer input tensor or shape

    axis: list or tuple of ints
        the axis to normalize on. If using BN on a dense layer then
        axis should be [0] to normalize over the samples. If the layer
        if a convolutional layer with data format NCHW then axis should
        be [0, 2, 3] to normalize over the samples and spatial dimensions
        (commonly done)

    deterministic: bool or Tensor
        controlling the state of the layer

    const: float32 (optional)
        the constant used in the standard deviation renormalization

    beta1: flaot32 (optional)
        the parameter for the exponential moving average of the mean

    beta2: float32 (optional)
        the parameters for the exponential moving average of the std

    Returns
    -------

    output: the layer output with attributes given by the layer options

    """
    def __init__(self, input_or_shape, axis, deterministic, const=0.001,
                 beta1=0.99, beta2=0.99, W=numpy.ones, b=numpy.zeros):

        self.init_input(input_or_shape)

        self.beta1 = beta1
        self.beta2 = beta2
        self.const = const
        self.axis = axis
        self.deterministic = deterministic

        parameter_shape = [self.input.shape[i] if i not in axis else 1
                           for i in range(self.input.ndim)]
        self.W = T.Variable(W, shape=parameter_shape, dtype='float32',
                            trainable=True)
        self.add_variable(self.W)

        self.b = T.Variable(b, shape=parameter_shape, dtype='float32',
                            trainable=True)
        self.add_variable(self.b)

        super().__init__(self.forward(self.input))

    def forward(self, input, deterministic=None):

        if deterministic is None:
            deterministic = self.deterministic
        dirac = T.cast(deterministic, 'float32')

        mean = T.mean(input, self.axis, keepdims=True)
        var = T.var(input, self.axis, keepdims=True)
        if len(self.updates.keys()) == 0:
            self.avgmean, upm, step = T.ExponentialMovingAverage(
                mean, self.beta1)
            self.avgvar, upv, step = T.ExponentialMovingAverage(
                var, self.beta2, step=step, init=numpy.ones(
                    var.shape).astype('float32'))
            self.add_variable(self.avgmean)
            self.add_variable(self.avgvar)
            self.add_update(upm)
            self.add_update(upv)

        usemean = mean * (1 - dirac) + self.avgmean * dirac
        usevar = var * (1 - dirac) + self.avgvar * dirac
        return self.W * (input - usemean) / \
            (T.sqrt(usevar) + self.const) + self.b
