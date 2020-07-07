from symjax import tensor as T
from . import ops_nn as nn
from symjax.nn import initializers, schedules
import symjax
import numpy
import jax


def _is_shape(x):
    """utility function that checks if the input is a shape or not"""
    if not hasattr(x, "__len__"):
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


class Layer(T.Op):
    def __init__(self, *args, name=None, **kwargs):

        if name is None:
            name = self.__NAME__

        with symjax.Scope(name):
            output = self.forward(*args, **kwargs)
            super().__init__(output, 0, _jax_function=jax.numpy.add)

    def variables(self, trainable=True):
        if not hasattr(self, "_variables"):
            return []
        if trainable is not None:
            return [v for v in self._variables if v.trainable == trainable]
        else:
            return self._variables

    def init_input(self, input_or_shape):
        if _is_shape(input_or_shape):
            self.input = T.Placeholder(input_or_shape, "float32")
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
            dtype = "float32"
        try:
            tensor = tensor_or_func(shape=shape, dtype=dtype)
        except TypeError:
            tensor = tensor_or_func(shape=shape).astype(dtype)
        return tensor

    def create_variable(self, name, tensor_or_func, shape, trainable, dtype=None):
        if tensor_or_func is None:
            self.__dict__[name] = None
            return
        t = self.create_tensor(tensor_or_func, shape, dtype)

        if not trainable:
            self.__dict__[name] = t
        else:
            self.__dict__[name] = T.Variable(t, name=name, trainable=True)
            self.add_variable(self.__dict__[name])

    @property
    def updates(self):
        if not hasattr(self, "_updates"):
            self._updates = {}
        return self._updates

    def add_update(self, update):
        symjax.current_graph().add(update)

    def add_variable(self, variable):
        if not hasattr(self, "_variables"):
            self._variables = list()
        self._variables.append(variable)

    def forward(self):
        pass


class Identity(Layer):

    __NAME__ = "Identity"

    def forward(self, input):
        return input


class Upsample1D(Layer):

    __NAME__ = "Upsample1D"

    def forward(self, input, repeat, axis=-1, mode="constant", value=0.0):
        return T.interpolation.upsample_1d(
            input, repeat=repeat, axis=axis, mode=mode, value=value,
        )


class Upsample2D(Layer):

    __NAME__ = "Upsample2D"

    def forward(self, input, repeat, axis, mode="constant", value=0.0):
        p1 = T.upsample_1d(
            input, repeat=repeat[0], axis=axis[0], mode=mode, value=value,
        )
        p2 = T.upsample_1d(p1, repeat=repeat[1], axis=axis[1], mode=mode, value=value,)
        return p2


class Dense(Layer):
    """Fully-connected/Dense layer

    perform a dense matrix multiplication and bias shifting of the
    input
    """

    __NAME__ = "Dense"

    def forward(
        self,
        input,
        units,
        W=initializers.he,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
    ):

        self.create_variable(
            "W", W, (numpy.prod(input.shape[1:]), units), trainable=trainable_W,
        )
        self.create_variable("b", b, (units,), trainable=trainable_b)

        if numpy.prod(input.shape[1:]) != self.W.shape[0]:
            raise RuntimeError("input to Dense layer {} has different dim".format(self))

        if self.b is not None:
            return T.dot(T.flatten2d(input), self.W) + self.b
        else:
            return T.dot(T.flatten2d(input), self.W)


class Conv1D(Layer):
    """1-D (time) convolution

    """

    __NAME__ = "Conv1D"

    def forward(
        self,
        input,
        n_filters,
        filter_length,
        W=initializers.he,
        b=numpy.zeros,
        stride=1,
        padding="VALID",
        trainable_W=True,
        trainable_b=True,
        input_dilations=None,
        filter_dilations=None,
    ):

        if numpy.isscalar(input_dilations):
            input_dilations = (input_dilations,) * 2
        self.input_dilation = input_dilations
        self.filter_dilation = filter_dilations
        self.stride = stride
        self.padding = padding

        self.create_variable(
            "W", W, (n_filters, input.shape[1], filter_length), trainable=trainable_W,
        )
        self.create_variable("b", b, (n_filters,), trainable=trainable_b)
        conv = nn.convNd(
            input,
            self.W,
            strides=self.stride,
            padding=self.padding,
            input_dilation=self.input_dilation,
            filter_dilation=self.filter_dilation,
        )
        if hasattr(self, "b"):
            return conv + self.b[:, None]
        else:
            return conv


class Conv2DTranspose(Layer):
    """2-D (spatial) convolution

    """

    __NAME__ = "Conv2DTranspose"

    def forward(
        self,
        input_or_shape,
        n_filters,
        filter_shape,
        pad="VALID",
        strides=1,
        W=initializers.he,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
        transpose_W=True,
        filter_dilations=None,
    ):

        self.init_input(input_or_shape)
        self.transpose_W = transpose_W
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.pad = pad

        self.create_variable(
            "W",
            W,
            (input.shape[1], n_filters) + tuple(filter_shape),
            trainable=trainable_W,
        )
        self.create_variable("b", b, (n_filters,), trainable=trainable_b)

        conv = T.convNd_transpose(
            input,
            self.W,
            strides=self.strides,
            padding=self.pad,
            transpose_kernel=self.transpose_W,
            filter_dilation=self.filter_dilation,
        )

        return conv + self.b.reshape((-1, 1, 1))


class Conv2D(Layer):
    """2-D (spatial) convolution

    """

    __NAME__ = "Conv2D"

    def forward(
        self,
        input,
        n_filters,
        filter_shape,
        pad="VALID",
        strides=1,
        W=initializers.he,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
        input_dilations=None,
        filter_dilations=None,
    ):

        self.input_dilation = input_dilations
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.pad = pad

        self.create_variable(
            "W",
            W,
            (n_filters, input.shape[1]) + tuple(filter_shape),
            trainable=trainable_W,
        )
        self.create_variable("b", b, (n_filters,), trainable=trainable_b)

        conv = nn.convNd(
            input,
            self.W,
            strides=self.strides,
            padding=self.pad,
            input_dilation=self.input_dilation,
            filter_dilation=self.filter_dilation,
        )
        if self.b is not None:
            return conv + self.b.reshape((-1, 1, 1))
        else:
            return conv


class Pool1D(Layer):
    """2-D (spatial) pooling

    """

    __NAME__ = "Pool1D"

    def forward(self, input_or_shape, pool_shape, pool_type="MAX", strides=None):

        self.init_input(input_or_shape)
        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape)
        if strides is None:
            self.strides = self.pool_shape
        else:
            self.strides = (1, 1, strides)

        return T.poolNd(
            input, self.pool_shape, strides=self.strides, reducer=self.pool_type,
        )


class Pool2D(Layer):
    """2-D (spatial) pooling

    """

    __NAME__ = "Pool2D"

    def forward(self, input, pool_shape, pool_type="MAX", strides=None):

        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape[0], pool_shape[1])
        if strides is None:
            self.strides = self.pool_shape
        else:
            if hasattr(strides, "__len__"):
                self.strides = (1, 1, strides[0], strides[1])
            else:
                self.strides = (1, 1, strides, strides)

        return nn.poolNd(
            input, self.pool_shape, strides=self.strides, reducer=self.pool_type,
        )


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

    __NAME__ = "Dropout"

    def forward(self, input, p, deterministic, seed=None):

        self.p = p
        self.mask = T.random.bernoulli(shape=input.shape, p=p, seed=seed)

        return T.where(deterministic, input, self.mask * input)


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

    __NAME__ = "RandomFlip"

    def forward(self, input, p, axis, deterministic, seed=None):

        self.p = p
        self.axis = axis
        extra_dims = input.ndim - 1
        self.flip = T.random.bernoulli(
            shape=(input.shape[0],) + (1,) * extra_dims, p=p, seed=seed
        )

        dirac = T.cast(deterministic, "float32")

        flipped_input = T.where(self.flip, T.flip(input, axis), input)

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

    __NAME__ = "RandomCrop"

    def forward(self, input, crop_shape, deterministic, padding=0, seed=None):

        self.crop_shape = crop_shape
        # if given only a scalar
        if not hasattr(padding, "__len__"):
            self.pad_shape = [(padding, padding)] * (input.shape - 1)
        # else
        else:
            self.pad_shape = [
                (pad, pad) if not hasattr(pad, "__len__") else pad for pad in padding
            ]

        assert len(self.pad_shape) == len(self.crop_shape)
        assert len(self.pad_shape) == (len(input.shape) - 1)

        self.start_indices = list()
        self.fixed_indices = list()
        for i, (pad, dim, crop) in enumerate(
            zip(self.pad_shape, input.shape[1:], self.crop_shape)
        ):
            maxval = pad[0] + pad[1] + dim - crop
            assert maxval >= 0
            self.start_indices.append(
                T.random.randint(
                    minval=0,
                    maxval=maxval,
                    shape=(input.shape[0], 1),
                    dtype="int32",
                    seed=seed + i if seed is not None else seed,
                )
            )

            self.fixed_indices.append(
                T.ones((input.shape[0], 1), "int32") * (maxval // 2)
            )
        self.start_indices = T.concatenate(self.start_indices, 1)
        self.fixed_indices = T.concatenate(self.fixed_indices, 1)

        dirac = T.cast(deterministic, "float32")

        # pad the input
        pinput = T.pad(input, [(0, 0)] + self.pad_shape)

        routput = T.stack(
            [
                T.dynamic_slice(pinput[n], self.start_indices[n], self.crop_shape)
                for n in range(input.shape[0])
            ],
            0,
        )
        doutput = T.stack(
            [
                T.dynamic_slice(pinput[n], self.fixed_indices[n], self.crop_shape)
                for n in range(input.shape[0])
            ],
            0,
        )

        return doutput * dirac + (1 - dirac) * routput


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

    __NAME__ = "BatchNormalization"

    def forward(
        self,
        input,
        axis,
        deterministic,
        const=1e-4,
        beta1=0.9,
        beta2=0.9,
        W=T.ones,
        b=T.zeros,
        trainable_W=True,
        trainable_b=True,
    ):

        self.beta1 = beta1
        self.beta2 = beta2
        self.const = const
        self.axis = axis
        self.deterministic = deterministic

        parameter_shape = [
            input.shape[i] if i in axis else 1 for i in range(input.ndim)
        ]
        reduce_axes = [i for i in range(input.ndim) if i not in axis]

        self.create_variable("W", W, parameter_shape, trainable=trainable_W)
        self.create_variable("b", b, parameter_shape, trainable=trainable_b)

        input_mean = T.mean(input, reduce_axes, keepdims=True)
        input_inv_std = 1 / (T.std(input, reduce_axes, keepdims=True) + const)

        self.avg_mean = schedules.ExponentialMovingAverage(input_mean, beta1)[1]
        self.avg_inv_std = schedules.ExponentialMovingAverage(input_inv_std, beta2)[1]

        use_mean = T.where(deterministic, self.avg_mean, input_mean)
        use_inv_std = T.where(deterministic, self.avg_inv_std, input_inv_std)
        W = self.W or 1.0
        b = self.b if self.b is not None else 0.0
        return W * (input - use_mean) * use_inv_std + b


class RNN(Layer):

    __NAME__ = "BatchNormalization"

    @staticmethod
    def gate(h, x, W, H, b, sigma):
        ht = sigma(T.dot(x, W) + b + T.dot(h, H))
        return ht, ht

    def forward(
        self,
        sequence,
        init_h,
        units,
        W=initializers.he,
        H=initializers.he,
        b=T.zeros,
        trainable_W=True,
        trainable_H=True,
        trainable_b=True,
        activation=nn.sigmoid,
        only_last=False,
    ):

        self.create_variable("W", W, (sequence.shape[2], units), trainable=trainable_W)
        self.create_variable("H", H, (units, units), trainable=trainable_H)
        self.create_variable("b", b, (units), trainable=trainable_b)

        last, output = T.scan(
            lambda h, x, W, H, b: self.gate(h, x, W, H, b, activation),
            init=init_h,
            sequences=[sequence.transpose((1, 0, 2))],
            non_sequences=[self.W, self.H, self.b],
        )
        if only_last:
            return last
        else:
            return output.transpose((1, 0, 2))


class GRU(Layer):

    __NAME__ = "BatchNormalization"

    @staticmethod
    def full_gate(h, x, Wh, Uh, bh, Wz, Uz, bz, Wr, Ur, br, sigma, phi):
        zt = sigma(T.dot(x, Wz) + bz + T.dot(h, Uz))
        rt = sigma(T.dot(x, Wr) + br + T.dot(h, Ur))
        h_hat = phi(T.dot(x, Wh) + bh + T.dot(h * rt, Uh))
        ht = (1 - zt) * h + zt * h_hat
        return ht, ht

    @staticmethod
    def minimal_gate(h, x, Wh, Uh, bh, Wz, Uz, bz, sigma, phi):
        ft = sigma(T.dot(x, Wz) + bz + T.dot(h, Uz))
        h_hat = phi(T.dot(x, Wh) + bh + T.dot(h * ft, Uh))
        ht = (1 - ft) * h + ft * h_hat
        return ht, ht

    def forward(
        self,
        sequence,
        init_h,
        units,
        Wh=initializers.he,
        Uh=initializers.he,
        bh=T.zeros,
        Wz=initializers.he,
        Uz=initializers.he,
        bz=T.zeros,
        Wr=initializers.he,
        Ur=initializers.he,
        br=T.zeros,
        trainable_Wh=True,
        trainable_Uh=True,
        trainable_bh=True,
        trainable_Wz=True,
        trainable_Uz=True,
        trainable_bz=True,
        trainable_Wr=True,
        trainable_Ur=True,
        trainable_br=True,
        activation=nn.sigmoid,
        phi=T.tanh,
        only_last=False,
        gate="minimal",
    ):

        self.create_variable(
            "Wh", Wh, (sequence.shape[2], units), trainable=trainable_Wh
        )
        self.create_variable("Uh", Uh, (units, units), trainable=trainable_Uh)
        self.create_variable("bh", bh, (units), trainable=trainable_bh)

        self.create_variable(
            "Wz", Wz, (sequence.shape[2], units), trainable=trainable_Wz
        )
        self.create_variable("Uz", Uz, (units, units), trainable=trainable_Uz)
        self.create_variable("bz", bz, (units), trainable=trainable_bz)

        if gate == "full":
            self.create_variable(
                "Wr", Wr, (sequence.shape[2], units), trainable=trainable_Wr
            )
            self.create_variable("Ur", Ur, (units, units), trainable=trainable_Ur)
            self.create_variable("br", br, (units), trainable=trainable_br)

        if gate == "minimal":

            def fn(h, x, Wh, Uh, bh, Wz, Uz, bz):
                return self.minimal_gate(h, x, Wh, Uh, bh, Wz, Uz, bz, activation, phi)

        elif gate == "full":

            def fn(h, x, Wh, Uh, bh, Wz, Uz, bz):
                return self.full_gate(
                    h, x, Wh, Uh, bh, Wz, Uz, bz, Wr, Ur, br, activation, phi
                )

        last, output = T.scan(
            fn,
            init=init_h,
            sequences=[sequence.transpose((1, 0, 2))],
            non_sequences=[self.W, self.H, self.b],
        )
        if only_last:
            return last
        else:
            return output.transpose((1, 0, 2))
