import numpy as np


def constant(shape, value):
    return np.full(shape, value)


def uniform(shape, scale=0.05):
    """Sample uniform weights U(-scale, scale).

    Parameters
    ----------

    shape: tuple

    scale: float (default=0.05)
    """
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05):
    """Sample Gaussian weights N(0, scale).

    Parameters
    ----------

    shape: tuple

    scale: float (default=0.05)

    """
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def orthogonal(shape, gain=1):
    """ From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[: shape[0], : shape[1]]


def get_fans(shape):
    """
    in all cases the fan_in is shape[0], and the fan_out is prod(shape[1:])
    """
    if len(shape) == 2:
        fan_in, fan_out = shape
    else:
        fan_out = shape[0]
        fan_in = np.prod(shape[1:])
    return fan_in, fan_out


def variance_scaling(shape, mode, gain=1, distribution=normal):
    """Variance Scaling initialization.
    """

    if len(shape) < 2:
        raise RuntimeError("This initializer only works with shapes of length >= 2")

    fan_in, fan_out = get_fans(shape)
    if mode == "fan_in":
        den = fan_in
    elif mode == "fan_out":
        den = fan_out
    elif mode == "fan_avg":
        den = (fan_in + fan_out) / 2.0
    elif mode == "fan_sum":
        den = fan_in + fan_out
    else:
        raise ValueError(
            "mode must be fan_in, fan_out, fan_avg or fan_sum, value passed was {mode}"
        )
    scale = gain * np.sqrt(1.0 / den)
    return distribution(shape, scale=scale)


def glorot_uniform(shape):
    """ Reference: Glorot & Bengio, AISTATS 2010
    """
    return variance_scaling(
        shape, mode="fan_sum", gain=np.sqrt(6), distribution=uniform
    )


def glorot_normal(shape):
    """ Reference: Glorot & Bengio, AISTATS 2010
    """
    return variance_scaling(shape, mode="fan_avg", distribution=normal)


def he_normal(shape):
    """ Reference:  He et al., http://arxiv.org/abs/1502.01852
    """
    return variance_scaling(shape, mode="fan_in", gain=np.sqrt(2), distribution=normal)


def he_uniform(shape):
    """ Reference:  He et al., http://arxiv.org/abs/1502.01852
    """
    return variance_scaling(shape, mode="fan_in", gain=np.sqrt(6), distribution=uniform)


def lecun_uniform(shape, name=None):
    """ Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    return variance_scaling(shape, mode="fan_in", gain=np.sqrt(3), distribution=uniform)
