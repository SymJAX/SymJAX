from symjax import tensor as T
from . import ops_nn as nn
from symjax.nn import initializers, schedules
import symjax
import numpy
import jax


class Layer(T.Op):
    # def __new__(cls, *args, name=None, **kwargs):

    #     if name is None:
    #         name = cls.__NAME__

    #     with symjax.Scope(name):
    #         output = cls.forward(*args, **kwargs)
    #     return output

    # def __init__(self, *args, _jax_function, _shapes, _dtypes, name=None, **kwargs):
    #     if name is None:
    #         name = _jax_function.__name__

    #     name, scope = symjax.current_graph()._get_name_scope(name, self)
    #     value = only_involves_shapes_or_constants(args)
    #     if len(kwargs):
    #         value = value * only_involves_shapes_or_constants(list(kwargs.values()))

    def __init__(self, *args, name=None, **kwargs):

        if name is None:
            name = self.__NAME__

        with symjax.Scope(name):
            output = self.forward(*args, **kwargs)
            super().__init__(
                output,
                0,
                _shape=symjax.current_graph().get_shape_dtype(output).shape,
                _dtype=output.dtype,
                _jax_function=jax.numpy.add,
            )

    def create_variable(self, name, tensor_or_func, shape, trainable, dtype="float32"):
        if tensor_or_func is None:
            self.__dict__[name] = None
            return

        self.__dict__[name] = T.Variable(
            tensor_or_func,
            name=name,
            shape=symjax.current_graph().get(shape),
            dtype=dtype,
            trainable=trainable,
        )

    def add_updates(self, update):
        symjax.current_graph().add_updates(update)

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
            input,
            repeat=repeat,
            axis=axis,
            mode=mode,
            value=value,
        )


class Upsample2D(Layer):

    __NAME__ = "Upsample2D"

    def forward(self, input, repeat, axis, mode="constant", value=0.0):
        p1 = T.upsample_1d(
            input,
            repeat=repeat[0],
            axis=axis[0],
            mode=mode,
            value=value,
        )
        p2 = T.upsample_1d(
            p1,
            repeat=repeat[1],
            axis=axis[1],
            mode=mode,
            value=value,
        )
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
        W=initializers.glorot_uniform,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
        flatten=True,
    ):
        if flatten:
            width_in = numpy.prod(input.shape[1:])
        else:
            width_in = input.shape[-1]

        self.create_variable(
            "W",
            W,
            (width_in, units),
            trainable=trainable_W,
        )
        self.create_variable("b", b, (units,), trainable=trainable_b)

        if flatten:
            flat_input = T.flatten2d(input)
        else:
            flat_input = input
        if self.b is not None:
            return T.dot(flat_input, self.W) + self.b
        else:
            return T.dot(flat_input, self.W)


class Conv1D(Layer):
    """1-D (time) convolution"""

    __NAME__ = "Conv1D"

    def forward(
        self,
        input,
        n_filters,
        filter_length,
        W=initializers.glorot_uniform,
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
            "W",
            W,
            (n_filters, input.shape[1], filter_length),
            trainable=trainable_W,
        )
        self.create_variable("b", b, (n_filters,), trainable=trainable_b)
        conv = T.signal.batch_convolve(
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
    """2-D (spatial) convolution"""

    __NAME__ = "Conv2DTranspose"

    def forward(
        self,
        input_or_shape,
        n_filters,
        filter_shape,
        pad="VALID",
        strides=1,
        W=initializers.glorot_uniform,
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

        conv = T.signal.batch_convolve_transpose(
            input,
            self.W,
            strides=self.strides,
            padding=self.pad,
            transpose_kernel=self.transpose_W,
            filter_dilation=self.filter_dilation,
        )

        return conv + self.b.reshape((-1, 1, 1))


class Conv2D(Layer):
    """2-D (spatial) convolution"""

    __NAME__ = "Conv2D"

    def forward(
        self,
        input,
        n_filters,
        filter_shape,
        pad="VALID",
        strides=1,
        W=initializers.glorot_uniform,
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

        conv = T.signal.batch_convolve(
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
    """2-D (spatial) pooling"""

    __NAME__ = "Pool1D"

    def forward(self, input_or_shape, pool_shape, pool_type="MAX", strides=None):

        self.init_input(input_or_shape)
        self.pool_type = pool_type
        self.pool_shape = (1, 1, pool_shape)
        if strides is None:
            self.strides = self.pool_shape
        else:
            self.strides = (1, 1, strides)

        return T.signal.pool(
            input,
            self.pool_shape,
            strides=self.strides,
            reducer=self.pool_type,
        )


class Pool2D(Layer):
    """2-D (spatial) pooling"""

    __NAME__ = "Pool2D"

    def forward(self, input, pool_shape, pool_type="MAX", strides=None):

        self.pool_type = pool_type
        self.pool_shape = (1, 1) + symjax.data.utils.as_tuple(pool_shape, 2)
        if strides is None:
            self.strides = self.pool_shape
        else:
            self.strides = (1, 1) + symjax.data.utils.as_tuple(strides, 2)

        return T.signal.pool(
            input,
            self.pool_shape,
            strides=self.strides,
            reducer=self.pool_type,
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
        self.mask = T.random.bernoulli(
            shape=symjax.current_graph().get(input.shape), p=p, seed=seed
        )

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
            shape=symjax.current_graph().get((input.shape[0],) + (1,) * extra_dims),
            p=p,
            seed=seed,
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
            self.pad_shape = [(padding, padding)] * (input.ndim - 1)
        # else
        else:
            self.pad_shape = [
                (pad, pad) if not hasattr(pad, "__len__") else pad for pad in padding
            ]

        assert len(self.pad_shape) == len(self.crop_shape)
        assert len(self.pad_shape) == input.ndim - 1

        self.start_indices = list()
        self.fixed_indices = list()
        for i, (pad, dim, crop) in enumerate(
            zip(self.pad_shape, input.shape[1:], self.crop_shape)
        ):
            maxval = pad[0] + pad[1] + dim - crop
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

        routput = T.map(
            lambda x, indices: T.dynamic_slice(x, indices, self.crop_shape),
            sequences=[pinput, self.start_indices],
        )
        doutput = T.map(
            lambda x, indices: T.dynamic_slice(x, indices, self.crop_shape),
            sequences=[pinput, self.fixed_indices],
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
        const=0.001,
        beta_1=0.99,
        beta_2=0.99,
        W=T.ones,
        b=T.zeros,
        trainable_W=True,
        trainable_b=True,
    ):

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.const = const
        self.axis = axis
        self.deterministic = deterministic

        parameter_shape = [
            input.shape[i] if i in axis else 1 for i in range(input.ndim)
        ]
        r_axes = [i for i in range(input.ndim) if i not in axis]

        self.create_variable("W", W, parameter_shape, trainable=trainable_W)
        self.create_variable("b", b, parameter_shape, trainable=trainable_b)

        self.input_mean = input.mean(r_axes, keepdims=True)
        # this definition is traditionally seen as less accurate than jnp.var's
        # mean((x - mean(x))**2) but may be faster and even, given typical
        # activation distributions and low-precision arithmetic, more accurate
        # when used in neural network normalization layers
        self.input_var = (
            (input ** 2).mean(r_axes, keepdims=True) - self.input_mean ** 2 + self.const
        )
        self.input_var = input.var(r_axes, keepdims=True)

        self.avg_mean = schedules.ExponentialMovingAverage(
            self.input_mean,
            beta_1,
            debias=False,
        )[1]
        self.avg_var = schedules.ExponentialMovingAverage(
            self.input_var,
            beta_2,
            init=T.ones_like(self.input_var, detach=True),
            debias=False,
        )[1]

        W = self.W if self.W is not None else 1.0
        b = self.b if self.b is not None else 0.0

        m = T.where(deterministic, self.avg_mean, self.input_mean)
        v = T.where(deterministic, self.avg_var, self.input_var)
        output = nn.normalize(input, mean=m, variance=v, epsilon=self.const)
        return W * output + b


class RNN(Layer):

    __NAME__ = "RNN"

    @staticmethod
    def gate(h, x, W, H, b, sigma):
        ht = sigma(T.dot(x, W) + b + T.dot(h, H))
        return ht, ht

    def forward(
        self,
        sequence,
        init_h,
        units,
        W=initializers.glorot_uniform,
        H=initializers.orthogonal,
        b=T.zeros,
        trainable_W=True,
        trainable_H=True,
        trainable_b=True,
        activation=nn.sigmoid,
        only_last=False,
    ):

        self.create_variable("W", W, (sequence.shape[2], units), trainable=trainable_W)
        self.create_variable("H", H, (units, units), trainable=trainable_H)
        self.create_variable("b", b, (units,), trainable=trainable_b)

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

    __NAME__ = "GRU"

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
        Wh=initializers.glorot_uniform,
        Uh=initializers.orthogonal,
        bh=T.zeros,
        Wz=initializers.glorot_uniform,
        Uz=initializers.orthogonal,
        bz=T.zeros,
        Wr=initializers.glorot_uniform,
        Ur=initializers.orthogonal,
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
        self.create_variable("bh", bh, (units,), trainable=trainable_bh)

        self.create_variable(
            "Wz", Wz, (sequence.shape[2], units), trainable=trainable_Wz
        )
        self.create_variable("Uz", Uz, (units, units), trainable=trainable_Uz)
        self.create_variable("bz", bz, (units,), trainable=trainable_bz)

        if gate == "full":
            self.create_variable(
                "Wr", Wr, (sequence.shape[2], units), trainable=trainable_Wr
            )
            self.create_variable("Ur", Ur, (units, units), trainable=trainable_Ur)
            self.create_variable("br", br, (units,), trainable=trainable_br)

        if gate == "minimal":

            def fn(*args):
                return self.minimal_gate(*args, activation, phi)

            last, output = T.scan(
                fn,
                init=init_h,
                sequences=[sequence.transpose((1, 0, 2))],
                non_sequences=[
                    self.Wh,
                    self.Uh,
                    self.bh,
                    self.Wz,
                    self.Uz,
                    self.bz,
                ],
            )

        elif gate == "full":

            def fn(*args):
                return self.full_gate(*args, activation, phi)

            last, output = T.scan(
                fn,
                init=init_h,
                sequences=[sequence.transpose((1, 0, 2))],
                non_sequences=[
                    self.Wh,
                    self.Uh,
                    self.bh,
                    self.Wz,
                    self.Uz,
                    self.bz,
                    self.Wr,
                    self.Ur,
                    self.br,
                ],
            )

        if only_last:
            return last
        else:
            return output.transpose((1, 0, 2))


class LSTM(Layer):

    __NAME__ = "GRU"

    @staticmethod
    def gate(
        carry,
        x,
        Wf,
        Uf,
        bf,
        Wi,
        Ui,
        bi,
        Wo,
        Uo,
        bo,
        Wc,
        Uc,
        bc,
        sigma_g,
        sigma_c,
        sigma_h,
    ):
        h, c = carry[0], carry[1]
        f = sigma_g(T.dot(x, Wf) + bf + T.dot(h, Uf))
        i = sigma_g(T.dot(x, Wi) + bi + T.dot(h, Ui))
        o = sigma_g(T.dot(x, Wo) + bo + T.dot(h, Uo))
        ctilde = sigma_c(T.dot(x, Wc) + bc + T.dot(h, Uc))
        cnew = f * c + i * ctilde
        hnew = o * sigma_h(cnew)
        return T.stack([hnew, cnew]), h

    def forward(
        self,
        sequence,
        init_h,
        units,
        Wf=initializers.glorot_uniform,
        Uf=initializers.orthogonal,
        bf=T.zeros,
        Wi=initializers.glorot_uniform,
        Ui=initializers.orthogonal,
        bi=T.zeros,
        Wo=initializers.glorot_uniform,
        Uo=initializers.orthogonal,
        bo=T.zeros,
        Wc=initializers.glorot_uniform,
        Uc=initializers.orthogonal,
        bc=T.zeros,
        trainable_Wf=True,
        trainable_Uf=True,
        trainable_bf=True,
        trainable_Wi=True,
        trainable_Ui=True,
        trainable_bi=True,
        trainable_Wo=True,
        trainable_Uo=True,
        trainable_bo=True,
        trainable_Wc=True,
        trainable_Uc=True,
        trainable_bc=True,
        activation_g=nn.sigmoid,
        activation_c=T.tanh,
        activation_h=T.tanh,
        only_last=False,
        gate="minimal",
    ):

        self.create_variable(
            "Wf", Wf, (sequence.shape[2], units), trainable=trainable_Wf
        )
        self.create_variable("Uf", Uf, (units, units), trainable=trainable_Uf)
        self.create_variable("bf", bf, (units,), trainable=trainable_bf)

        self.create_variable(
            "Wi", Wi, (sequence.shape[2], units), trainable=trainable_Wi
        )
        self.create_variable("Ui", Ui, (units, units), trainable=trainable_Ui)
        self.create_variable("bi", bi, (units,), trainable=trainable_bi)

        self.create_variable(
            "Wo", Wo, (sequence.shape[2], units), trainable=trainable_Wo
        )
        self.create_variable("Uo", Uo, (units, units), trainable=trainable_Uo)
        self.create_variable("bo", bo, (units,), trainable=trainable_bo)

        self.create_variable(
            "Wc", Wc, (sequence.shape[2], units), trainable=trainable_Wc
        )
        self.create_variable("Uc", Uc, (units, units), trainable=trainable_Uc)
        self.create_variable("bc", bc, (units,), trainable=trainable_bc)

        def fn(*args):
            return self.gate(*args, activation_g, activation_c, activation_h)

        init = T.stack((init_h, T.zeros(init_h.shape, init_h.dtype)))
        last, output = T.scan(
            fn,
            init=init,
            sequences=[sequence.transpose((1, 0, 2))],
            non_sequences=[
                self.Wf,
                self.Uf,
                self.bf,
                self.Wi,
                self.Ui,
                self.bi,
                self.Wo,
                self.Uo,
                self.bo,
                self.Wc,
                self.Uc,
                self.bc,
            ],
        )

        if only_last:
            return last
        else:
            return output.transpose((1, 0, 2))
