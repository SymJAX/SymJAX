import jax.random as jnp
import jax
from .base import RandomOp, jax_wrap
from .ops_math import cast




_RANDOM_FUNCTIONS = [jax.random.bernoulli, jax.random.beta, jax.random.cauchy,
                     jax.random.dirichlet, jax.random.gamma, jax.random.gumbel,
                     jax.random.laplace, jax.random.logit,
                     jax.random.multivariate_normal, jax.random.normal,
                     jax.random.pareto, jax.random.randint, jax.random.shuffle,
                     jax.random.threefry_2x32, jax.random.truncated_normal,
                     jax.random.uniform]




#_normal = RandomOp(jnp.normal)
#_uniform = RandomOp(jnp.uniform)
#_bernoulli = RandomOp(jnp.bernoulli)

randn = jax_wrap(jnp.normal)

#def permutation(n, dtype='int32'):
#    output = jax.numpy.arange(n).astype(dtype)
#    return shuffle(output)
#
#
#
#class _normal(RandomOp):
#    @staticmethod
#    def fn(key, shape, dtype='float32'):
#        return jnp.normal(key=key, shape=shape, dtype=dtype)
#
#class uniform(RandomOp):
#    @staticmethod
#    def fn(key, shape, dtype='float32', minval=0., maxval=1.):
#        return jnp.uniform(key=key, shape=shape, dtype=dtype, minval=minval,
#                           maxval=maxval)
#
#class shuffle(RandomOp):
#    @staticmethod
#    def fn(key, x, axis=0):
#        return jnp.shuffle(key=key, x=x, axis=axis)
#
#
#class randint(RandomOp):
#    @staticmethod
#    def fn(key, minval, maxval, shape, dtype='int32'):
#        return jnp.randint(key=key, shape=shape, dtype=dtype, minval=minval,
#                           maxval=maxval)
#
#class bernoulli(RandomOp):
#    @staticmethod
#    def fn(key, p, shape):
#        return jnp.bernoulli(key=key, p=p, shape=shape)
#
#
#
#def normal(shape=(), mean=0, var=1, dtype='float32'):
#        """
#        Generate a random Normal tensor with i.i.d. elements with
#        given mean and variance.
#
#        Parameters:
#        -----------
#
#        shape : (tuple)
#            the shape of the tensor to be generated
#        mean : scalar (optional, default=0)
#            the mean of the Normal distribution
#        var : scalar (optional, defualt=1)
#            the variance of the Normal distribution
#
#        Returns:
#        --------
#
#        tensor : RandomTensor
#            the generated random variable object
#        """
#
#        return _normal(shape=shape, dtype=dtype)*var+mean
#
#
#randn = normal
#rand = uniform
#
##def uniform(shape=(), lower=0, upper=1, dtype='float32'):
##    """
##    Generate a Uniform random tensor with i.i.d. elements with
##    given upper and lower bound.
##
##    Parameters:
##    -----------
##
##    shape : (tuple)
##        the shape of the tensor to be generated
##
##    lower : scalar (optional, default=0)
##        the lower bound of the distribution
##
##    upper : scalar (optional, defualt=1)
##        the upper bound of the distribution
##
##    Returns:
##    --------
##
##    tensor : RandomTensor
##        the generated random variable object
##    """
##    return _uniform(shape=shape, minval=lower, maxval=upper, dtype=dtype, _dtype=dtype, _shape=shape)
##
##
##def bernoulli(shape=(), p=0.5, dtype='bool', name=''):
##    """
##    Generate a Bernoulli random tensor with i.i.d. elements with
##    success probability.
##
##    Args:
##    -----
##        shape (:obj:`tuple`): the shape of the tensor to be generated
##        p (scalar, optional): success probability. Defaults to 0.5
##
##    Returns:
##    --------
##        RandomTensor: the generated random variable object
##
##    Examples:
##    ---------
##        >>> print(bernoulli((3, 3), 0.5))
##        (RandomTensor Bernoulli(0.2): dtype=bool, shape=(3, 3))
##    """
##    rv = _bernoulli(shape=shape, p=p, name=name,
##                    descr='Bernoulli(' + str(p) + ')')
##    if dtype == 'bool':
##        return rv
##    else:
##        return cast(rv, dtype)
##
##
