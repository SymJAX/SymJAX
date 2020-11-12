from symjax import tensor as T
from . import ops_nn as nn
from symjax.nn import initializers, schedules
import symjax
import numpy
import jax


# IMPORTANT NOTE
# in order to make sphinx doc clean we use a hacky way to use the
# __init__ method as a staticmethod wich has an actual return ...
# not pythonic, suggestions welcome !


def create_variable(
    name,
    tensor_or_func,
    shape,
    trainable,
    inplace=False,
    dtype="float32",
    preprocessor=None,
):
    if tensor_or_func is None:
        return None

    if inplace:
        assert not callable(tensor_or_func)
        return tensor_or_func

    variable = T.Variable(
        tensor_or_func,
        name=name,
        shape=symjax.current_graph().get(shape),
        dtype=dtype,
        trainable=trainable,
    )
    if preprocessor is not None:
        return preprocessor(variable)
    else:
        return variable


class Layer(T.Tensor):
    def __new__(cls, *args, name=None, **kwargs):

        if name is None:
            name = cls.__NAME__

        with symjax.Scope(name):

            output = cls.__init__(cls, *args, **kwargs)

        return output

    @staticmethod
    def add_updates(self, update):
        symjax.current_graph().add_updates(update)

    def forward(self):
        pass


class Identity(Layer):

    __NAME__ = "Identity"

    def __init__(self, input):
        return input


class Upsample1D(Layer):

    __NAME__ = "Upsample1D"

    def __init__(self, input, repeat, axis=-1, mode="constant", value=0.0):
        return T.interpolation.upsample_1d(
            input,
            repeat=repeat,
            axis=axis,
            mode=mode,
            value=value,
        )


class Upsample2D(Layer):

    __NAME__ = "Upsample2D"

    def __init__(self, input, repeat, axis, mode="constant", value=0.0):
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

    Parameters:
    -----------
    input: Tensor
        the input to the layer (does not have to be 2D)

    units: int
        the width of the layer

    W: Tensor-like/ndarray or callable (default initializers.glorot_uniform)
        the matrix weight of the layer of shape (units, input_dim)

    b: Tensor-like/ndarray or callable (default numpy.zeros)
        the bias vector of the layer

    trainable_W: bool (default True)
        if the variable initialized from W should be trainable

    trainable_b: bool (default True)
        if the vector initialized from b should be trainable

    W_preprocessor: None or callable (default: None)
        a possible preprocessing function applied onto the layer variable of W
        before computing the layer output

    b_preprocessor:  None or callable (default: None)
        a possible preprocessing function applied onto the layer variable of b
        before computing the layer output

    inplace_W: bool (default False)
        if the given Tensor-like/array or callable W should be used in place and
        not put as a Variable (which is then either frozen or learned) This is
        useful to create multiple layers with same weights.

    inplace_b: bool (default False)
        same as inplace_W but for b

    flatten: bool (default True)
        whether to flatten or not the input if more than 2 dimensional

    """

    __NAME__ = "Dense"

    def __init__(
        self,
        input,
        units,
        W=initializers.glorot_uniform,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
        W_preprocessor=None,
        b_preprocessor=None,
        inplace_W=False,
        inplace_b=False,
        flatten=True,
    ):
        if flatten:
            width_in = numpy.prod(input.shape[1:])
        else:
            width_in = input.shape[-1]

        W = create_variable(
            "W",
            W,
            (units, width_in),
            trainable=trainable_W,
            preprocessor=W_preprocessor,
            inplace=inplace_W,
        )

        b = create_variable(
            "b",
            b,
            (units,),
            trainable=trainable_b,
            preprocessor=b_preprocessor,
            inplace=inplace_b,
        )

        if flatten:
            flat_input = T.flatten2d(input)
        else:
            flat_input = input
        if b is not None and W is None:
            return flat_input + b
        elif b is None and W is not None:
            return T.dot(flat_input, W.T)
        elif b is not None and W is not None:
            return T.dot(flat_input, W.T) + b
        else:
            return flat_input


class Conv1D(Layer):
    """1-D (time) convolution

    perform a dense matrix multiplication and bias shifting of the
    input

    Parameters:
    -----------

    input

    n_filters

    filter_length

    W=initializers.glorot_uniform

    b=numpy.zeros

    stride=1

    padding="VALID"

    trainable_W=True

    trainable_b=True

    inplace_W=False

    inplace_b=False

    W_preprocessor=None

    b_preprocessor=None

    input_dilations=None

    filter_dilations=None


    """

    __NAME__ = "Conv1D"

    def __init__(
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
        inplace_W=False,
        inplace_b=False,
        W_preprocessor=None,
        b_preprocessor=None,
        input_dilations=None,
        filter_dilations=None,
    ):

        if numpy.isscalar(input_dilations):
            input_dilations = (input_dilations,) * 2

        W = create_variable(
            "W",
            W,
            (n_filters, input.shape[1], filter_length),
            trainable=trainable_W,
            preprocessor=W_preprocessor,
            inplace=inplace_W,
        )
        b = create_variable(
            "b",
            b,
            (n_filters,),
            trainable=trainable_b,
            preprocessor=b_preprocessor,
            inplace=inplace_b,
        )
        conv = T.signal.batch_convolve(
            input,
            W,
            strides=stride,
            padding=padding,
            input_dilation=input_dilations,
            filter_dilation=filter_dilations,
        )
        if b is not None:
            return conv + b[:, None]
        else:
            return conv


class Conv2DTranspose(Layer):
    """2-D (spatial) convolution"""

    __NAME__ = "Conv2DTranspose"

    def __init__(
        self,
        input,
        n_filters,
        filter_shape,
        padding="VALID",
        strides=1,
        W=initializers.glorot_uniform,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
        transpose_W=True,
        filter_dilations=None,
    ):

        self.init_input(input)
        self.transpose_W = transpose_W
        self.filter_dilation = filter_dilations
        self.strides = strides
        self.padding = padding

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
            padding=self.padding,
            transpose_kernel=self.transpose_W,
            filter_dilation=self.filter_dilation,
        )

        return conv + self.b.reshape((-1, 1, 1))


class Conv2D(Layer):
    """2-D (spatial) convolution"""

    __NAME__ = "Conv2D"

    def __init__(
        self,
        input,
        n_filters,
        filter_shape,
        padding="VALID",
        strides=1,
        W=initializers.glorot_uniform,
        b=numpy.zeros,
        trainable_W=True,
        trainable_b=True,
        inplace_W=False,
        inplace_b=False,
        input_dilations=None,
        filter_dilations=None,
        W_preprocessor=None,
        b_preprocessor=None,
    ):

        W = create_variable(
            "W",
            W,
            (n_filters, input.shape[1]) + tuple(filter_shape),
            trainable=trainable_W,
            preprocessor=W_preprocessor,
            inplace=inplace_W,
        )
        b = create_variable(
            "b",
            b,
            (n_filters,),
            trainable=trainable_b,
            preprocessor=b_preprocessor,
            inplace=inplace_b,
        )

        conv = T.signal.batch_convolve(
            input,
            W,
            strides=strides,
            padding=padding,
            input_dilation=input_dilations,
            filter_dilation=filter_dilations,
        )
        if b is not None:
            return conv + b.reshape((-1, 1, 1))
        else:
            return conv


class Pool1D(Layer):
    """2-D (spatial) pooling"""

    __NAME__ = "Pool1D"

    def __init__(self, input, pool_shape, pool_type="MAX", strides=None):

        pool_shape = (1, 1, pool_shape)
        if strides is None:
            strides = pool_shape
        else:
            strides = (1, 1, strides)

        return T.signal.pool(
            input,
            pool_shape,
            strides=strides,
            reducer=pool_type,
        )


class Pool2D(Layer):
    """2-D (spatial) pooling"""

    __NAME__ = "Pool2D"

    def __init__(self, input, pool_shape, pool_type="MAX", strides=None):

        pool_shape = (1, 1) + symjax.data.utils.as_tuple(pool_shape, 2)
        if strides is None:
            strides = pool_shape
        else:
            strides = (1, 1) + symjax.data.utils.as_tuple(strides, 2)

        return T.signal.pool(
            input,
            pool_shape,
            strides=strides,
            reducer=pool_type,
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

    def __init__(self, input, p, deterministic, seed=None):

        mask = T.random.bernoulli(shape=input.shape, p=p, seed=seed)

        return T.where(deterministic, input, mask * input)


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

    def __init__(self, input, p, axis, deterministic, seed=None):

        extra_dims = input.ndim - 1
        flip = T.random.bernoulli(
            shape=(input.shape[0],) + (1,) * extra_dims,
            p=p,
            seed=seed,
        )

        dirac = T.cast(deterministic, "float32")

        flipped_input = T.where(flip, T.flip(input, axis), input)

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

    def __init__(self, input, crop_shape, deterministic, padding=0, seed=None):

        # if given only a scalar
        if not hasattr(padding, "__len__"):
            pad_shape = [(padding, padding)] * (input.ndim - 1)
        # else
        else:
            pad_shape = [
                (pad, pad) if not hasattr(pad, "__len__") else pad for pad in padding
            ]

        assert len(pad_shape) == len(crop_shape)
        assert len(pad_shape) == input.ndim - 1

        start_indices = list()
        fixed_indices = list()
        for i, (pad, dim, crop) in enumerate(
            zip(pad_shape, input.shape[1:], crop_shape)
        ):
            maxval = pad[0] + pad[1] + dim - crop
            start_indices.append(
                T.random.randint(
                    minval=0,
                    maxval=maxval,
                    shape=(input.shape[0], 1),
                    dtype="int32",
                    seed=seed + i if seed is not None else seed,
                )
            )

            fixed_indices.append(T.ones((input.shape[0], 1), "int32") * (maxval // 2))
        start_indices = T.concatenate(start_indices, 1)
        fixed_indices = T.concatenate(fixed_indices, 1)

        dirac = T.cast(deterministic, "float32")

        # pad the input
        pinput = T.pad(input, [(0, 0)] + pad_shape)

        routput = T.map(
            lambda x, indices: T.dynamic_slice(x, indices, crop_shape),
            sequences=[pinput, start_indices],
        )
        doutput = T.map(
            lambda x, indices: T.dynamic_slice(x, indices, crop_shape),
            sequences=[pinput, fixed_indices],
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

    def __init__(
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

        parameter_shape = [
            input.shape[i] if i in axis else 1 for i in range(input.ndim)
        ]
        r_axes = [i for i in range(input.ndim) if i not in axis]

        W = create_variable("W", W, parameter_shape, trainable=trainable_W)
        b = create_variable("b", b, parameter_shape, trainable=trainable_b)

        input_mean = input.mean(r_axes, keepdims=True)
        # this definition is traditionally seen as less accurate than jnp.var's
        # mean((x - mean(x))**2) but may be faster and even, given typical
        # activation distributions and low-precision arithmetic, more accurate
        # when used in neural network normalization layers
        input_var = (input ** 2).mean(r_axes, keepdims=True) - input_mean ** 2 + const
        input_var = input.var(r_axes, keepdims=True)

        avg_mean = schedules.ExponentialMovingAverage(
            input_mean, beta_1, debias=False, name="mean_ema"
        )[1]
        avg_var = schedules.ExponentialMovingAverage(
            input_var,
            beta_2,
            init=T.ones_like(input_var, detach=True),
            debias=False,
            name="var_ema",
        )[1]

        m = T.where(deterministic, avg_mean, input_mean)
        v = T.where(deterministic, avg_var, input_var)
        output = nn.normalize(input, mean=m, variance=v, epsilon=const)
        if b is None and W is not None:
            return W * output
        elif b is not None and W is None:
            return output + b
        elif b is not None and W is not None:
            return W * output + b
        else:
            return output


class RNN(Layer):

    __NAME__ = "RNN"

    @staticmethod
    def gate(h, x, W, H, b, sigma):
        ht = sigma(T.dot(x, W) + b + T.dot(h, H))
        return ht, ht

    def __init__(
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

        W = create_variable("W", W, (sequence.shape[2], units), trainable=trainable_W)
        H = create_variable("H", H, (units, units), trainable=trainable_H)
        b = create_variable("b", b, (units,), trainable=trainable_b)

        last, output = T.scan(
            lambda h, x, W, H, b: RNN.gate(h, x, W, H, b, activation),
            init=init_h,
            sequences=[sequence.transpose((1, 0, 2))],
            non_sequences=[W, H, b],
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

    def __init__(
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

        Wh = create_variable(
            "Wh", Wh, (sequence.shape[2], units), trainable=trainable_Wh
        )
        Uh = create_variable("Uh", Uh, (units, units), trainable=trainable_Uh)
        bh = create_variable("bh", bh, (units,), trainable=trainable_bh)

        Wz = create_variable(
            "Wz", Wz, (sequence.shape[2], units), trainable=trainable_Wz
        )
        Uz = create_variable("Uz", Uz, (units, units), trainable=trainable_Uz)
        bz = create_variable("bz", bz, (units,), trainable=trainable_bz)

        if gate == "full":
            Wr = create_variable(
                "Wr", Wr, (sequence.shape[2], units), trainable=trainable_Wr
            )
            Ur = create_variable("Ur", Ur, (units, units), trainable=trainable_Ur)
            br = create_variable("br", br, (units,), trainable=trainable_br)

        if gate == "minimal":

            def fn(*args):
                return GRU.minimal_gate(*args, activation, phi)

            last, output = T.scan(
                fn,
                init=init_h,
                sequences=[sequence.transpose((1, 0, 2))],
                non_sequences=[Wh, Uh, bh, Wz, Uz, bz],
            )

        elif gate == "full":

            def fn(*args):
                return GRU.full_gate(*args, activation, phi)

            last, output = T.scan(
                fn,
                init=init_h,
                sequences=[sequence.transpose((1, 0, 2))],
                non_sequences=[Wh, Uh, bh, Wz, Uz, bz, Wr, Ur, br],
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

    def __init__(
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
