from symjax.tensor import ops_methods
from symjax import tensor as T
from symjax import initializers
#from symjax import get_graph
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

def get_variables(layers, trainable=True):
    return sum([l.variables(trainable) for l in layers if hasattr(l, 'variables')], [])

def get_updates(layers):
    updates = dict()
    for l in layers:
        if hasattr(l, 'updates'):
            updates.update(l.updates)
    return updates


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

    def create_tensor(self, tensor_or_func, shape, dtype=None):
        if not callable(tensor_or_func):
            if tensor_or_func is None:
                return None
            assert tensor_or_func.shape == shape
            if tensor_or_func.dtype != dtype and dtype is not None:
                return tensor_or_func.astype(dtype)
            else:
                return tensor_or_func
        if dtype is None:
            dtype = 'float32'
        try:
            tensor = tensor_or_func(shape=shape, dtype=dtype)
        except:
            tensor = tensor_or_func(shape=shape).astype(dtype)
        return tensor

    def create_variable(self, name, tensor_or_func, shape, trainable,
                        dtype=None):
        if tensor_or_func is None:
            return None
        t = self.create_tensor(tensor_or_func, shape, dtype)

        if not trainable:
            self.__dict__[name] = t
        else:
            self.__dict__[name] = T.Variable(t, name=name, trainable=True)
            self.add_variable(self.__dict__[name])


    @property
    def updates(self):
        if not hasattr(self, '_updates'):
            self._updates = {}
        return self._updates

    def add_update(self, update):
        self._updates.update(update)
#        if get_graph() is not None:
#            get_graph().updates.update(update)

    def add_variable(self, variable):
        if not hasattr(self, '_variables'):
            self._variables = list()
        self._variables.append(variable)

    def forward(self):
        pass


class Identity(Layer):
    name = 'Identity'
    def __init__(self, input_or_shape):

        self.init_input(input_or_shape)
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return input


class Upsample1D(Layer):
    name = 'Upsample1D'
    def __init__(self, input_or_shape, repeat, axis=-1, mode='constant',
                 value=0.):

        self.init_input(input_or_shape)
        self.repeat = repeat
        self.mode = mode
        self.value = value
        self.axis = axis
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return T.upsample_1d(input, repeat=self.repeat, axis=self.axis,
                             mode=self.mode, value=self.value)


class Upsample2D(Layer):
    name = 'Upsample2D'
    def __init__(self, input_or_shape, repeat, axis, mode='constant',
                 value=0.):

        self.init_input(input_or_shape)
        self.repeat = repeat
        self.mode = mode
        self.value = value
        self.axis = axis
        super().__init__(self.forward(self.input))

    def forward(self, input):
        p1 = T.upsample_1d(input, repeat=self.repeat[0], axis=self.axis[0],
                             mode=self.mode, value=self.value)
        p2 = T.upsample_1d(p1, repeat=self.repeat[1], axis=self.axis[1],
                             mode=self.mode, value=self.value)
        return p2


class Reshape(Layer):
    name = 'Reshape'
    def __init__(self, input_or_shape, shape):

        self.init_input(input_or_shape)
        self.new_shape = shape
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return T.reshape(input, self.new_shape)


class Upsample(Layer):
    name = 'Upsample'
    def __init__(self, input_or_shape, factors):

        self.init_input(input_or_shape)
        self.factors = factors
        super().__init__(self.forward(self.input))

    def forward(self, input):
        upsample = T.upsample(input, self.factors)
        return upsample


class Lambda(Layer):
    """lambda function applied onto the input

    applies a function (such as activation function) onto the input.
    """
    name = 'Lambda'
    def __init__(self, input_or_shape, fn):

        self.init_input(input_or_shape)
        self.fn = fn
        super().__init__(self.forward(self.input))

    def forward(self, input):
        return self.fn(input)


class Dense(Layer):
    """Fully-connected/Dense layer

    perform a dense matrix multiplication and bias shifting of the
    input
    """
    name = 'Dense'
    def __init__(self, input_or_shape, units, W=initializers.he,
                 b=numpy.zeros, trainable_W=True, trainable_b=True):

        self.init_input(input_or_shape)

        self.create_variable('W', W, (numpy.prod(self.input.shape[1:]), units),
                            trainable=trainable_W)
        self.create_variable('b', b, (units,), trainable=trainable_b)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        if numpy.prod(input.shape[1:]) != self.W.shape[0]:
            raise RuntimeError(
                'input to Dense layer {} has different dim'.format(self))
        if hasattr(self, 'b'):
            return T.dot(T.flatten2d(input), self.W) + self.b
        else:
            return T.dot(T.flatten2d(input), self.W)


class Conv1D(Layer):
    """1-D (time) convolution

    """
    name = 'Conv1D'
    def __init__(self, input_or_shape, n_filters, filter_length,
                 W=initializers.he, b=numpy.zeros,
                 stride=1, pad='VALID', trainable_W=True, trainable_b=True,
                 input_dilations=None, filter_dilations=None):

        self.init_input(input_or_shape)
        if numpy.isscalar(input_dilations):
            input_dilations = (input_dilations,) * 2
        self.input_dilation = input_dilations
        self.filter_dilation = filter_dilations
        self.stride = stride
        self.pad = pad

        self.create_variable('W', W, (n_filters, self.input.shape[1],
                                   filter_length), trainable=trainable_W)
        self.create_variable('b', b, (n_filters,), trainable=trainable_b)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        conv = T.convNd(input, self.W, strides=self.stride, padding=self.pad,
                        input_dilation=self.input_dilation,
                        filter_dilation=self.filter_dilation)
        if hasattr(self, 'b'):
            return conv + self.b[:, None]
        else:
            return conv


class Conv2DTranspose(Layer):
    """2-D (spatial) convolution

    """
    name = 'Conv2DTranspose'
    def __init__(self, input_or_shape, n_filters, filter_shape, pad='VALID',
                 strides=1, W=initializers.he, b=numpy.zeros,
                 trainable_W=True, trainable_b=True, transpose_W=True,
                 filter_dilations=None):

        self.init_input(input_or_shape)
        self.transpose_W = transpose_W
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.pad = pad

        self.create_variable('W', W,
                        (self.input.shape[1], n_filters) + tuple(filter_shape),
                            trainable=trainable_W)
        self.create_variable('b', b, (n_filters,), trainable=trainable_b)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        conv = T.convNd_transpose(input, self.W, strides=self.strides, padding=self.pad,
                        transpose_kernel=self.transpose_W,
                        filter_dilation=self.filter_dilation)

        return conv + self.b.reshape((-1, 1, 1))



class Conv2D(Layer):
    """2-D (spatial) convolution

    """
    name = 'Conv2D'
    def __init__(self, input_or_shape, n_filters, filter_shape, pad='VALID',
                 strides=1, W=initializers.he, b=numpy.zeros,
                 trainable_W=True, trainable_b=True,
                 input_dilations=None, filter_dilations=None):

        self.init_input(input_or_shape)
        self.input_dilation = input_dilations
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.pad = pad

        self.create_variable('W', W,
                        (n_filters, self.input.shape[1]) + tuple(filter_shape),
                            trainable=trainable_W)
        self.create_variable('b', b, (n_filters,), trainable=trainable_b)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        conv = T.convNd(input, self.W, strides=self.strides, padding=self.pad,
                        input_dilation=self.input_dilation,
                        filter_dilation=self.filter_dilation)
        if hasattr(self, 'b'):
            return conv + self.b.reshape((-1, 1, 1))
        else:
            return conv


class Pool1D(Layer):
    """2-D (spatial) pooling

    """
    name = 'Pool1D'
    def __init__(self, input_or_shape, pool_shape, pool_type='MAX',
                 strides=None):

        self.init_input(input_or_shape)
        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape)
        if strides is None:
            self.strides = self.pool_shape
        else:
            self.strides = (1, 1, strides)

        super().__init__(self.forward(self.input))

    def forward(self, input):
        return T.poolNd(input, self.pool_shape, strides=self.strides,
                        reducer=self.pool_type)


class Pool2D(Layer):
    """2-D (spatial) pooling

    """
    name = 'Pool2D'
    def __init__(self, input_or_shape, pool_shape, pool_type='MAX',
                 strides=None):

        self.init_input(input_or_shape)
        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape[0], pool_shape[1])
        if strides is None:
            self.strides = self.pool_shape
        else:
            if hasattr(strides, '__len__'):
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
    name = 'Dropout'
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
    name = 'RandomFlip'
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
    name = 'RandomCrop'
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
    name = 'BatchNormalization'
    def __init__(self, input_or_shape, axis, deterministic, const=0.001,
                 beta1=0.99, beta2=0.99, W=numpy.ones, b=numpy.zeros,
                 trainable_W=True, trainable_b=True):

        self.init_input(input_or_shape)

        self.beta1 = beta1
        self.beta2 = beta2
        self.const = const
        self.axis = axis
        self.deterministic = deterministic

        parameter_shape = [self.input.shape[i] if i not in axis else 1
                           for i in range(self.input.ndim)]

        self.create_variable('W', W, parameter_shape, trainable=trainable_W)
        self.create_variable('b', b, parameter_shape, trainable=trainable_b)

        super().__init__(self.forward(self.input))

    def forward(self, input, deterministic=None):

        if deterministic is None:
            deterministic = self.deterministic
        dirac = T.cast(deterministic, 'float32')

        self.mean = T.mean(input, self.axis, keepdims=True)
        self.var = T.var(input, self.axis, keepdims=True)
        if len(self.updates) == 0:
            self.avgmean, upm, step = T.ExponentialMovingAverage(
                self.mean, self.beta1)
            self.avgvar, upv, step = T.ExponentialMovingAverage(
                self.var, self.beta2, step=step, init=numpy.ones(
                    self.var.shape).astype('float32'))
            self.add_variable(self.avgmean)
            self.add_variable(self.avgvar)
            self.add_update(upm)
            self.add_update(upv)

        self.usemean = self.mean * (1 - dirac) + self.avgmean * dirac
        self.usevar = self.var * (1 - dirac) + self.avgvar * dirac
        return self.W * (input - self.usemean) / \
            (T.sqrt(self.usevar) + self.const) + self.b
