import jax.random as jnp
from .base import RandomOp
from .ops_base import cast

_normal = RandomOp(jnp.normal)
_uniform = RandomOp(jnp.uniform)
_bernoulli = RandomOp(jnp.bernoulli)

def normal(shape=(), mean=0, var=1, dtype='float32'):
    """
    Generate a random Normal tensor with i.i.d. elements with
    given mean and variance.

    Parameters:
    -----------

    shape : (tuple)
        the shape of the tensor to be generated
    mean : scalar (optional, default=0)
        the mean of the Normal distribution
    var : scalar (optional, defualt=1)
        the variance of the Normal distribution

    Returns:
    --------

    tensor : RandomTensor
        the generated random variable object
    """

    return _normal(shape=shape, dtype=dtype, _dtype=dtype, _shape=shape)*var+mean


def uniform(shape=(), lower=0, upper=1, dtype='float32'):
    """
    Generate a Uniform random tensor with i.i.d. elements with
    given upper and lower bound.

    Parameters:
    -----------

    shape : (tuple)
        the shape of the tensor to be generated

    lower : scalar (optional, default=0)
        the lower bound of the distribution

    upper : scalar (optional, defualt=1)
        the upper bound of the distribution

    Returns:
    --------

    tensor : RandomTensor
        the generated random variable object
    """
    return _uniform(shape=shape, minval=lower, maxval=upper, dtype=dtype, _dtype=dtype, _shape=shape)


def bernoulli(shape=(), p=0.5, dtype='bool', name=''):
    """
    Generate a Bernoulli random tensor with i.i.d. elements with
    success probability.

    Args:
    -----
        shape (:obj:`tuple`): the shape of the tensor to be generated
        p (scalar, optional): success probability. Defaults to 0.5

    Returns:
    --------
        RandomTensor: the generated random variable object

    Examples:
    ---------
        >>> print(bernoulli((3, 3), 0.5))
        (RandomTensor Bernoulli(0.2): dtype=bool, shape=(3, 3))
    """
    rv = _bernoulli(shape=shape, p=p, name=name,
                    descr='Bernoulli(' + str(p) + ')')
    if dtype == 'bool':
        return rv
    else:
        return cast(rv, dtype)


