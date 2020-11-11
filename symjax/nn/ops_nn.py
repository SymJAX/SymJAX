import symjax.tensor as T


def relu(x):
    r"""Rectified linear unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{relu}(x) = \max(x, 0)
    """
    return T.maximum(x, 0)


def softplus(x):
    r"""Softplus activation function.

    Computes the element-wise function

    .. math::
      \mathrm{softplus}(x) = \log(1 + e^x)
    """
    return T.logaddexp(x, 0)


def soft_sign(x):
    r"""Soft-sign activation function.

    Computes the element-wise function

    .. math::
      \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}
    """
    return x / (T.abs(x) + 1)


def sigmoid(x):
    r"""Sigmoid activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}
    """
    return T.expit(x)


def silu(x):
    r"""SiLU activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}
    """
    return x * T.sigmoid(x)


def swish(x, beta):
    r"""Swish activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-\beta * x}}
    """
    return x * T.sigmoid(beta * x)


def leaky_swish(x, beta, negative_slope=1e-2):
    r"""Swish activation function associated to leaky relu.

    Computes the element-wise function:

    .. math::
      \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-\beta * x}}
    """

    feature = T.stack([negative_slope * x, x], -1)
    return (feature * softmax(feature * beta)).sum(-1)


def log_sigmoid(x):
    r"""Log-sigmoid activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})
    """
    return -softplus(-x)


def elu(x, alpha=1.0):
    r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}
  """
    safe_x = T.where(x > 0, 0.0, x)
    return T.where(x > 0, x, alpha * T.expm1(safe_x))


def leaky_relu(x, negative_slope=1e-2):
    r"""Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{leaky\_relu}(x) = \begin{cases}
      x, & x \ge 0\\
      \alpha x, & x < 0
    \end{cases}

  where :math:`\alpha` = :code:`negative_slope`.
  """
    return T.where(x >= 0, x, negative_slope * x)


def hard_tanh(x):
    r"""Hard :math:`\mathrm{tanh}` activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & 0 \le x \le 1\\
      1, & 1 < x
    \end{cases}
  """
    return T.where(x > 1, 1, T.where(x < -1, -1, x))


def celu(x, alpha=1.0):
    r"""Continuously-differentiable exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{celu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Continuously Differentiable Exponential Linear Units
  <https://arxiv.org/pdf/1704.07483.pdf>`_."""
    return T.where(x > 0, x, alpha * T.expm1(x / alpha))


def selu(x):
    r"""Scaled exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{selu}(x) = \lambda \begin{cases}
      x, & x > 0\\
      \alpha e^x - \alpha, & x \le 0
    \end{cases}

  where :math:`\lambda = 1.0507009873554804934193349852946` and
  :math:`\alpha = 1.6732632423543772848170429916717`.

  For more information, see
  `Self-Normalizing Neural Networks
  <https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf>`_.
  """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)


def gelu(x, approximate: bool = True):
    r"""Gaussian error linear unit activation function.

    If ``approximate=False``, computes the element-wise function:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
        \frac{x}{\sqrt{2}} \right) \right)

    If ``approximate=True``, uses the approximate formulation of GELU:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
        \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    For more information, see `Gaussian Error Linear Units (GELUs)
    <https://arxiv.org/abs/1606.08415>`_, section 2.

    Args:
      approximate: whether to use the approximate or exact formulation.
    """
    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
        cdf = 0.5 * (1.0 + T.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
        return x * cdf
    else:
        raise NotImplemented
        # return x * (lax.erf(x / np.sqrt(2)) + 1) / 2, dtype=x.dtype)


def glu(linear_x, gated_x, axis=-1):
    """Gated linear unit activation function."""

    return linear_x * sigmoid(gated_x)


def log_softmax(x, axis=-1):
    r"""Log-Softmax function.

    Computes the logarithm of the :code:`softmax` function, which rescales
    elements to the range :math:`[-\infty, 0)`.

    .. math ::
      \mathrm{log\_softmax}(x) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
      \right)

    Args:
      axis: the axis or axes along which the :code:`log_softmax` should be
        computed. Either an integer or a tuple of integers.
    """
    shifted = x - T.stop_gradient(x.max(axis, keepdims=True))
    return shifted - T.log(T.sum(T.exp(shifted), axis, keepdims=True))


def softmax(x, axis=-1):
    r"""Softmax function.

    Computes the function which rescales elements to the range :math:`[0, 1]`
    such that the elements along :code:`axis` sum to :math:`1`.

    .. math ::
      \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Args:
      axis: the axis or axes along which the softmax should be computed. The
        softmax output summed across these dimensions should sum to :math:`1`.
        Either an integer or a tuple of integers.
    """
    unnormalized = T.exp(x - T.stop_gradient(x.max(axis, keepdims=True)))
    return unnormalized / unnormalized.sum(axis, keepdims=True)


def normalize(x, axis=-1, mean=None, variance=None, epsilon=1e-5):
    """Normalizes an array by subtracting mean and dividing by sqrt(var)."""
    if mean is None:
        mean = T.mean(x, axis, keepdims=True)
    if variance is None:
        # this definition is traditionally seen as less accurate than jnp.var's
        # mean((x - mean(x))**2) but may be faster and even, given typical
        # activation distributions and low-precision arithmetic, more accurate
        # when used in neural network normalization layers
        variance = T.mean(T.square(x), axis, keepdims=True) - T.square(mean)
    return (x - mean) * T.rsqrt(variance + epsilon)


def relu6(x):
    r"""Rectified Linear Unit 6 activation function.

    Computes the element-wise function

    .. math::
      \mathrm{relu6}(x) = \min(\max(x, 0), 6)
    """
    return T.minimum(T.maximum(x, 0), 6.0)


def hard_sigmoid(x):
    r"""Hard Sigmoid activation function.

    Computes the element-wise function

    .. math::
      \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}
    """
    return relu6(x + 3.0) / 6.0


def hard_silu(x):
    r"""Hard SiLU activation function

    Computes the element-wise function

    .. math::
      \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)
    """
    return x * hard_sigmoid(x)


def log_1_minus_sigmoid(x):
    return -module.__dict__["softplus"](x)
