from . import tensor as T
from . import initializers
import numpy
import inspect


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
        self._variables = list()
        self.init_input(input)
        self.init_variables(**kwargs)

        # now we create the default arguments that will be used to compute the
        # default output value of the layer/tensor.
        signature = list(inspect.signature(self.forward).parameters.keys())
        self.default_kwargs = dict()
        for name in signature[1:]:
            self.default_kwargs.update({name: self.__dict__[name]})

        # we can now compute the default output value
        output = self.forward(input, **self.default_kwargs)
        super().__init__(output.shape, output.dtype, output.roots, copyof=output)

    def variables(self, trainable=True):
        print(self._variables)
        return [v for v in self._variables if v.trainable==trainable]

    def init_variables(self, **kwargs):
        pass

    def init_input(self, input_or_shape):
        if is_shape(input_or_shape):
            self.layer_input = T.Placeholder(input_or_shape, 'float32')
        else:
            self.layer_input = input_or_shape
        self.input_shape = self.layer_input.shape

    def add_variable(self, variable):
        self._variables.append(variable)

    def forward(self):
        pass


#class Identity(Layer):
#    def __init__(self, input_or_shape):
#        super().__init__(input_or_shape)
#
#    def forward(self, input, **kwargs):
#        return input




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
            self.W = T.Variable(W((numpy.prod(self.input_shape[1:]), units)))
            self.add_variable(self.W)
        elif W is None:
            self.W = 1
        else:
            assert W.shape == (numpy.prod(self.input_shape[1:]), units)
            self.W = W

        if callable(b):
            self.b = T.Variable(b((units,)))
            self.add_variable(self.b)
        elif b is None:
            self.b = 0
        else:
            assert b.shape == (units,)
            self.b = b

    @staticmethod
    def forward(input, W, b):
        if numpy.prod(input.shape[1:]) != W.shape[0]:
            raise RuntimeError('input to Dense layer {} has different dim'.format(self))
        if input.ndim > 2:
            output = T.dot(T.flatten2d(input), W) + b
        else:
            output = T.dot(input, W) + b
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
                 W=initializers.he, b=numpy.zeros, mode='valid'):
        super().__init__(input_or_shape, n_filters=n_filters,
                         filter_shape=filter_shape, strides=strides,
                         W=W, b=b, mode=mode)

    def init_variables(self, n_filters, filter_shape, W, b, strides, mode,**kwargs):
        self.strides = strides
        self.mode = mode
        if callable(W):
            self.W = T.Variable(W((n_filters, self.input_shape[1], filter_shape[0],
                                filter_shape[1])))
            self.add_variable(self.W)
        elif W is None:
            self.W = 1
        else:
            assert W.shape == (n_filters, self.input_shape[1], filter_shape[0],
                              filter_shape[1])
            self.W = W
        if callable(b):
            self.b = T.Variable(b((n_filters,)))
            self.add_variable(self.b)
        elif b is None:
            self.b = 0
        else:
            assert b.shape == (n_filters,)
            self.b = b


    @staticmethod
    def forward(input, W, b, strides, mode):
        if not numpy.isscalar(b):
            if b.ndim == 1:
                b2 = T.expand_dims(T.expand_dims(b, 1), 1)
            else:
                b2 = b
        else:
            b2 = b
        spatial = W.shape[2:]
        if mode == 'same':
            h, w = (spatial[0]-1) // 2, (spatial[1]-1) // 2
            padding = [(h, spatial[0]-1 - h), (w, spatial[1] -1 - w)]
            input2 = T.pad(input, [(0, 0), (0, 0), padding[0], padding[1]])
        elif mode == 'full':
            h, w = spatial[0]-1 , spatial[1]-1
            padding = [(h, h), (w, w)]
            input2 = T.pad(input, [(0, 0), (0, 0), padding[0], padding[1]])
        else:
            input2 = input
        output = T.convNd(input2, W, strides=strides) + b2
        return output


class Pool2D(Layer):
    def __init__(self, input_or_shape, pool_shape, pool_type='MAX',
                 strides=None):
        super().__init__(input_or_shape, pool_shape=pool_shape,
                         strides=strides, pool_type=pool_type)

    def init_variables(self, pool_shape, strides, pool_type, **kwargs):
        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape[0], pool_shape[1])
        if strides is None:
            self.strides = self.pool_shape
        else:
            if hasattr(strides, __len__):
                self.strides = (1, 1, strides[0], strides[1])
            else:
                self.strides = (1, 1, strides, strides)

    @staticmethod
    def forward(input, pool_shape, strides, pool_type):
        output = T.poolNd(input, pool_shape, strides=strides,
                          reducer=pool_type)
        return output



class Dropout(Layer):

    def __init__(self, input_or_shape, p, deterministic, seed=None):
        super().__init__(input_or_shape, p=p, deterministic=deterministic,
                         seed=seed)

    def init_variables(self, p, deterministic, seed):
        self.deterministic = deterministic
        self.p = p
        self.seed = seed

    def forward(self, input, p, deterministic, seed):
        dirac = T.cast(deterministic, 'float32')
        if seed is None:
            seed = numpy.random.randint(0, 100000)
        if not hasattr(self, 'mask'):
            self.mask = T.random.bernoulli(shape=input.shape, p=p, seed=seed)
        return  input * self.mask * (1 - dirac) + input * dirac



class RandomCrop(Layer):

    def __init__(self, input_or_shape, crop_shape, pad_shape, deterministic,
                 seed=None):
        super().__init__(input_or_shape, crop_shape=crop_shape, pad_shape=pad_shape,
                         deterministic=deterministic, seed=seed)

    def init_variables(self, crop_shape, pad_shape, deterministic, seed):
        self.crop_shape = crop_shape
        self.pad_shape = pad_shape
        self.deterministic = deterministic
        self.seed = seed

    def forward(self, input, crop_shape, pad_shape, deterministic, seed):
        dirac = T.cast(deterministic, 'float32')
        # adjust the pad_shape if needed
        if not hasattr(pad_shape, '__len__'):
            pad_shape = [(pad_shape, pad_shape), (pad_shape, pad_shape)]
        elif not hasattr(pad_shape[0], '__len__'):
            pad_shape = [(pad_shape[0], pad_shape[0]), (pad_shape[1],
                                                        pad_shape[1])]

        # pad the input as needed and flatten it
        if pad_shape[0][0] > 0 or pad_shape[0][1] > 0\
            or pad_shape[1][0] > 0 or pad_shape[1][1] > 0:
            input2 = T.pad(input, [(0, 0), (0, 0), pad_shape[0], pad_shape[1]])
        else:
            input2 = input
        flat_input = input2.reshape(input2.shape[:2] + (-1,))

        # compute the base indices of a 2d patch
        patch = T.arange(numpy.prod(crop_shape)).reshape(crop_shape)
        offset = T.expand_dims(T.arange(crop_shape[0]), 1)
        patch_indices = patch + offset * (input2.shape[3] - crop_shape[1])
        # flatten them
        flat_indices = patch_indices.reshape((1, 1, -1))
        # and repeat for the input channels
        cflat_indices = flat_indices#.repeat(input.shape[1], 1)

        # create the random shifts and get the random patch indices
        if seed is None:
            seed = numpy.random.randint(0,100000)
        if not hasattr(self, 'h_ind'):
            h = input2.shape[2] - crop_shape[0]
            self.h_ind = T.random.randint(minval=0, maxval=h + 1,
                                          shape=(input.shape[0],),
                                          dtype='int32', seed=seed)
            w = input2.shape[3] - crop_shape[1]
            self.w_ind = T.random.randint(minval=0, maxval=w + 1,
                                          shape=(input.shape[0],),
                                          dtype='int32', seed=seed+1)

        random_offsets = self.h_ind * input2.shape[3] + self.w_ind
        crandom_offsets = random_offsets.reshape((-1, 1, 1))
        random_indices = cflat_indices + crandom_offsets
        flat_output = T.take_along_axis(flat_input, random_indices, 2)
        output = flat_output.reshape(input.shape[:2] + tuple(crop_shape))

        # create the deterministic version
        if crop_shape[0] != input.shape[2] or crop_shape[1] != input.shape[3]:
            offset = (pad_shape[0][0] + (input.shape[2] - crop_shape[0])//2,
                      pad_shape[1][0] + (input.shape[3] - crop_shape[1])//2)
            offset = input2.shape[3] * offset[0] + offset[1]
            deter_output = T.take_along_axis(flat_input, flat_indices + offset, 2).reshape(input.shape[:2] + tuple(crop_shape))
        else:
            deter_output = input
        return output#deter_output * dirac +  (1 - dirac) * output






class BatchNormalization(Layer):
    def __init__(self, input_or_shape, axis, deterministic, const=0.001,
                 beta1=0.99, beta2=0.99, W=numpy.ones, b=numpy.zeros):
        super().__init__(input_or_shape, axis=axis,
                         deterministic=deterministic, const=const,
                         beta1=beta1, beta2=beta2, W=W, b=b)

    def init_variables(self, axis, W, b, beta1, beta2, const, deterministic, 
                       **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.const = const
        self.axis = axis
        self.deterministic = deterministic
        parameter_shape = [self.input_shape[i] if i not in axis else 1
                           for i in range(len(self.input_shape))]
        if callable(W):
            self.W = T.Variable(W(parameter_shape))
            self.add_variable(self.W)
        elif W is None:
            self.W = 1
        else:
            assert W.shape == parameter_shape
            self.W = W
        if callable(b):
            self.b = T.Variable(b(parameter_shape))
            self.add_variable(self.b)
        elif b is None:
            self.b = 0
        else:
            assert b.shape == parameter_shape
            self.b = b

    def forward(self, input, deterministic, axis, beta1, beta2, const):
        mean = T.mean(input, axis, keepdims=True)
        var = T.var(input, axis, keepdims=True)
        if len(self.updates.keys()) == 0:
#            self.beta1, self.beta2 = beta1, beta2
            self.avgmean, upm, step = T.ExponentialMovingAverage(mean, beta1)
            self.avgvar, upv, step = T.ExponentialMovingAverage(var, beta2, step=step, init=numpy.ones(var.shape))
            self.add_variable(self.avgmean)
            self.add_variable(self.avgvar)
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
