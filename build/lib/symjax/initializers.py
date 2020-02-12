import numpy


def constant(shape, value):
    return numpy.full(shape, value)

def uniform(shape, range=0.01, std=None, mean=0.):
    """Sample initial weights from the uniform distribution.
    Parameters are sampled from U(a, b).
    Parameters
    ----------
    range : float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    std : float or None
        If std is a float then the weights are sampled from
        U(mean - numpy.sqrt(3) * std, mean + numpy.sqrt(3) * std).
    mean : float
        see std for description.
    """
    if std is not None:
        a = mean - numpy.sqrt(3) * std
        b = mean + numpy.sqrt(3) * std
    elif hasattr(range, '__len__'):
        a, b = range  # range is a tuple
    else:
        a, b = -range, range  # range is a number
    return numpy.random.rand(*shape) * (b - a) + a

def normal(shape, mean=0., std=1.):
    """Sample initial weights from the Gaussian distribution.
    Initial weight parameters are sampled from N(mean, std).
    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    return numpy.random.randn(*shape)*std + mean


def orthogonal(shape, gain=1, seed=None):
     flat_shape = (shape[0], numpy.prod(shape[1:]))
     a = numpy.random.normal(flat_shape, seed=seed)
     u, _, v = numpy.linalg.svd(a, full_matrices=False)
     # pick the one with the correct shape
     q = u if u.shape == flat_shape else v
     q = q.reshape(shape)
     return gain * q


def glorot(shape, gain=1, distribution=normal):
    """Glorot weight initialization.
    This is also known as Xavier initialization [1]_.
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
    """
    if len(shape) < 2:
        raise RuntimeError(
                     "This initializer only works with shapes of length >= 2")

    n1, n2 = shape[:2]
    receptive_field_size = numpy.prod(shape[2:])
    std = gain * numpy.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    return distribution(shape, std=std)


def he(shape, gain=numpy.sqrt(2), distribution=normal):
    """He weight initialization.
    Weights are initialized with a standard deviation of
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
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) > 2:
        fan_in = numpy.prod(shape[1:])
    std = gain * numpy.sqrt(1.0 / fan_in)
    return distribution(shape, std=std)





