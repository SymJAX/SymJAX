from . import tensor as T
from . import initializers
import numpy


def is_shape(x):
    if not hasattr(x, '__len__'):
        return False
    if type(x[0]) == int:
        return True
    elif type(x[0]) == float:
        raise RuntimeError("invalid float value in shape")





class Layer(T.Tensor):
    def __init__(self, input, **kwargs):
        self.updates = dict()
        self.variables = list()
        self.init_input(input)
        self.init_variables(**kwargs)
        output = self.forward(self.layer_input, **kwargs)
        super().__init__(output)

    def init_variables(self, **kwargs):
        pass

    def init_input(self, input_or_shape):
        if is_shape(input_or_shape):
            self.layer_input = T.Placeholder(input_or_shape, 'float32')
        else:
            self.layer_input = input_or_shape
        self.input_shape = self.layer_input.shape

    def add_variable(self, variable, name=None):
        self.variables.append(variable)
        if name is not None:
            self.__dict__[name] = variable




class Activation(Layer):
    def __init__(self, input_or_shape, activation):
        super().__init__(input_or_shape, activation=activation)

    def forward(self, input, **kwargs):
        return kwargs['activation'](input)




class Dense(Layer):
    def __init__(self, input_or_shape, units, W=initializers.he,
                 b=numpy.zeros):
        super().__init__(input_or_shape, units=units, W=W, b=b)

    def init_variables(self, units, W, b, **kwargs):
        if callable(W):
            W = T.Variable(W((numpy.prod(self.input_shape[1:]), units)))
            self.add_variable(W, 'W')
        else:
            assert W.shape == (numpy.prod(self.input_shape[1:]), units)
            self.W = W
        if callable(b):
            b = T.Variable(b((units,)))
            self.add_variable(b, 'b')
        else:
            assert b.shape == (units,)
            self.b = b

    def forward(self, input, **kwargs):
        if numpy.prod(input.shape[1:]) != self.W.shape[0]:
            raise RuntimeError('input to Dense layer {} has different dim'.format(self))
        if input.ndim > 2:
            output = T.dot(T.flatten2d(input), self.W) + self.b
        else:
            output = T.dot(input, self.W) + self.b
        return output



class Conv1D(Layer):

    def __init__(self, input_or_shape, n_filters, filter_length, strides=1,
                 W=initializers.he, b=numpy.zeros):
        super().__init__(input_or_shape, n_filters=n_filters,
                         filter_length=filter_length, strides=strides,
                         W=W, b=b)

    def init_variables(self, n_filters, filter_length, W, b, **kwargs):
        if callable(W):
            W = T.Variable(W((n_filters, self.input_shape[1], filter_length)))
            self.add_variable(W, 'W')
        else:
            assert W.shape == (n_filters, self.input_shape[1], filter_length)
            self.W = W
        if callable(b):
            b = T.Variable(b((n_filters,)))
            self.add_variable(b, 'b')
        else:
            assert b.shape == (n_filters,)
            self.b = b

    def forward(self, input, **kwargs):
        return T.convNd(input, self.W, strides=kwargs['strides']) + T.expand_dims(self.b, 1)

class Conv2D(Layer):
    def __init__(self, input_or_shape, n_filters, filter_shape, strides=1, 
                 W=initializers.he, b=numpy.zeros):
        super().__init__(input_or_shape, n_filters=n_filters,
                         filter_shape=filter_shape, strides=strides,
                         W=W, b=b)

    def init_variables(self, n_filters, filter_shape, W, b, **kwargs):
        if callable(W):
            W = T.Variable(W((n_filters, self.input_shape[1], filter_shape[0],
                              filter_shape[1])))
            self.add_variable(W, 'W')
        else:
            assert W.shape == (n_filters, self.input_shape[1], filter_shape[0],
                              filter_shape[1])
            self.W = W
        if callable(b):
            b = T.Variable(b((n_filters,)))
            self.add_variable(b, 'b')
        elif b is None:
            self.b = None
        else:
            assert b.shape == (n_filters,)
            self.b = b

    def forward(self, input, **kwargs):
        if self.b is None:
            b = 0
        else:
            b = T.expand_dims(T.expand_dims(self.b, 1), 1)
        output = T.convNd(input, self.W, strides=kwargs['strides']) + b
        return output

class Pool2D(Layer):
    def __init__(self, input_or_shape, pool_shape, pool_type='MAX',
                 strides=None):
        pool_shape = (1, 1, pool_shape[0], pool_shape[1])
        if strides is None:
            strides = pool_shape
        else:
            if hasattr(strides, __len__):
                strides = (1, 1, strides[0], strides[1])
            else:
                strides = (1, 1, strides, strides)
        super().__init__(input_or_shape, pool_shape=pool_shape,
                         strides=strides, pool_type=pool_type)

    def forward(self, input, **kwargs):
        output = T.poolNd(input, kwargs['pool_shape'],
                          strides=kwargs['strides'],
                          reducer=kwargs['pool_type'])
        return output



class Dropout(Layer):

    def __init__(self, input_or_shape, p, deterministic):
        super().__init__(input_or_shape, p=p, deterministic=deterministic)

    def forward(self, input, p, deterministic, **kwargs):
        dirac = T.cast(deterministic, 'float32')
        if not hasattr(self, 'mask'):
            self.p = p
            self.mask = T.random.bernoulli(self.input_shape, p=p, dtype='float32')
        else:
            assert self.p == p
        return (input * self.mask) * (1 - dirac) + input * dirac


class BatchNormalization(Layer):
    def __init__(self, input_or_shape, axis, deterministic, const=0.001,
                 beta1=0.99, beta2=0.99, W=numpy.ones, b=numpy.zeros):
        super().__init__(input_or_shape, axis=axis,
                         deterministic=deterministic, const=const,
                         beta1=beta1, beta2=beta2, W=W, b=b)

    def init_variables(self, axis, W, b, **kwargs):
        parameter_shape = [self.input_shape[i] if i not in axis else 1
                           for i in range(len(self.input_shape))]
        if callable(W):
            W = T.Variable(W(parameter_shape))
            self.add_variable(W, 'W')
        else:
            assert W.shape == parameter_shape
            self.W = W
        if callable(b):
            b = T.Variable(b(parameter_shape))
            self.add_variable(b, 'b')
        else:
            assert b.shape == parameter_shape
            self.b = b

    def forward(self, input, deterministic, axis, beta1, beta2, const, **kwargs):
        mean = T.mean(input, axis, keepdims=True)
        var = T.var(input, axis, keepdims=True)
        if len(self.updates.keys()) == 0:
            self.beta1, self.beta2 = beta1, beta2
            self.avgmean, upm, step = T.ExponentialMovingAverage(mean, beta1)
            self.avgvar, upv, step = T.ExponentialMovingAverage(var, beta2, step=step, init=numpy.ones(var.shape))
            self.updates.update({**upm, **upv})
        else:
            assert beta1 == self.beta1
            assert beta2 == self.beta2
        dirac = T.cast(deterministic, 'float32')
        usemean = mean * (1 - dirac) + self.avgmean * dirac
        usevar = var * (1 - dirac) + self.avgvar * dirac
        output = self.W * (input - usemean) / (T.sqrt(usevar) +\
                         const) + self.b
        return output
