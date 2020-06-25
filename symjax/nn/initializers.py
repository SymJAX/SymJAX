import numpy as np


def constant(shape, value):
    return np.full(shape, value)


def uniform(shape, range=0.01, std=None, mean=0.0):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).

    Parameters
    ----------

    range: float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).

    std: float or None
        If std is a float then the weights are sampled from
        U(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std).

    mean: float
        see std for description.
        :param shape:
    """
    if std is not None:
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
    elif hasattr(range, "__len__"):
        a, b = range  # range is a tuple
    else:
        a, b = -range, range  # range is a number
    return np.random.rand(*shape) * (b - a) + a


def normal(shape, mean=0.0, std=1.0):
    """Sample initial weights from the Gaussian distribution.

    Initial weight parameters are sampled from N(mean, std).

    Parameters
    ----------
    
    std: float
        Std of initial parameters.
    
    mean: float
        Mean of initial parameters.
        :param shape:

    """
    return np.random.randn(*shape) * std + mean


def orthogonal(shape, gain=1):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q


def _compute_fans(shape, in_axis=0, out_axis=1):
    print(shape)
    receptive_field_size = np.prod(shape) / (shape[in_axis] * shape[out_axis])
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out


def variance_scaling(mode, shape, gain, distribution=normal, in_axis=0, out_axis=1):
    """Variance Scaling initialization.
    """
    if len(shape) < 2:
        raise RuntimeError("This initializer only works with shapes of length >= 2")

    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
        den = fan_in
    elif mode == "fan_out":
        den = fan_out
    elif mode == "fan_avg":
        den = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(
            "mode must be fan_in, fan_out or fan_avg, value passed was {mode}"
        )
    std = gain * np.sqrt(1.0 / den)
    return distribution(shape, std=std)


def glorot(shape, gain=1, distribution=normal, in_axis=0, out_axis=1):
    """Glorot weight initialization.

    This is also known as Xavier initialization [1]_.

    Parameters
    ----------
    
    initializer: lasagne.init.Initializer
        Initializer used to sample the weights, must accept `std` in its
        constructor to sample from a distribution with a given standard
        deviation.
    
    gain: float or 'relu'
        Scaling factor for the weights. Set this to ``1.0`` for linear and
        sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
        to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
        leakiness ``alpha``. Other transfer functions may need different
        factors.
    
    c01b: bool
        For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
        with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
        the correct fan-in and fan-out.
    
    References
    ----------
    
    .. [1] Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    
    Notes
    -----
    
    For a :class:`DenseLayer <lasagne.layers.DenseLayer>`, if ``gain='relu'``
    and ``initializer=Uniform``, the weights are initialized as
    .. math::
       a &= \\sqrt{\\frac{12}{fan_{in}+fan_{out}}}\\\\
       W &\sim U[-a, a]
    If ``gain=1`` and ``initializer=Normal``, the weights are initialized as
    .. math::
       \\sigma &= \\sqrt{\\frac{2}{fan_{in}+fan_{out}}}\\\\
       W &\sim N(0, \\sigma)
       :param shape:
       :param distribution:
       :param in_axis:
       :param out_axis:
    """
    if len(shape) < 2:
        raise RuntimeError("This initializer only works with shapes of length >= 2")

    return variance_scaling(
        "fan_avg", shape, gain, distribution, in_axis=in_axis, out_axis=out_axis
    )


def he(shape, gain=np.sqrt(2), distribution=normal, in_axis=0, out_axis=1):
    """He weight initialization.
    Weights are initialized with a standard deviation of
    :param shape:
    :param distribution:
    :param in_axis:
    :param out_axis:
    :math:`\\sigma = gain \\sqrt{\\frac{1}{fan_{in}}}` [1]_.
    
    Parameters
    ----------
    
    initializer : lasagne.init.Initializer
        Initializer used to sample the weights, must accept `std` in its
        constructor to sample from a distribution with a given standard
        deviation.
    
    gain : float or 'relu'
        Scaling factor for the weights. Set this to ``1.0`` for linear and
        sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
        to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
        leakiness ``alpha``. Other transfer functions may need different
        factors.
    
    c01b : bool
        For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
        with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
        the correct fan-in and fan-out.
    
    References
    ----------
    
    .. [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.
    
    See Also
    ----------
    
    HeNormal  : Shortcut with Gaussian initializer.
    HeUniform : Shortcut with uniform initializer.
    """
    return variance_scaling(
        "fan_in", shape, gain, distribution, in_axis=in_axis, out_axis=out_axis
    )


def lecun(shape, gain=1.0, distribution=normal, in_axis=0, out_axis=1):
    """LeCun weight initialization.
    Weights are initialized with a standard deviation of
    :math:`\\sigma = gain \\sqrt{\\frac{1}{fan_{in}}}`.
   """
    return variance_scaling(
        "fan_in", shape, gain, distribution, in_axis=in_axis, out_axis=out_axis
    )
