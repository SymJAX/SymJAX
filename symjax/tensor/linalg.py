import inspect
import sys

import jax.numpy.linalg as jnpl
import jax.scipy.linalg as jspl

from .base import jax_wrap
from .normalization import normalize

from . import random
from . import ops_numpy as T

from_scipy = [
    "cholesky",
    "block_diag",
    "cho_solve",
    "eigh",
    "expm",
    # "expm_frechet",
    "inv",
    "lu",
    "lu_factor",
    "lu_solve",
    "solve_triangular",
    "tril",
    "triu",
]

NAMES = [c[0] for c in inspect.getmembers(jnpl, inspect.isfunction)] + [
    "pinv",
    "slogdet",
]
NAMES.remove("norm")

module = sys.modules[__name__]
for name in NAMES:
    if name not in from_scipy:
        module.__dict__.update({name: jax_wrap(jnpl.__dict__[name])})

for name in from_scipy:
    module.__dict__.update({name: jax_wrap(jspl.__dict__[name])})


_norm = jax_wrap(jnpl.__dict__["norm"])


def norm(x, ord=2, axis=None, keepdims=False):
    """
    Tensor/Matrix/Vector norm.

    For matrices and vectors,
    this function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    for higher-dimensional tensors, only :math:`0<ord<\\infty` is supported.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is `2`.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.
        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.
        .. versionadded:: 1.10.0
    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).
    See Also
    --------
    scipy.linalg.norm : Similar function in SciPy.
    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.
    The following norms can be calculated:
    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================
    The Frobenius norm is given by [1]_:
        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`
    The nuclear norm is the sum of the singular values.
    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.
    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, ...,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])
    >>> LA.norm(a)
    7.745966692414834
    >>> LA.norm(b)
    7.745966692414834
    >>> LA.norm(b, 'fro')
    7.745966692414834
    >>> LA.norm(a, np.inf)
    4.0
    >>> LA.norm(b, np.inf)
    9.0
    >>> LA.norm(a, -np.inf)
    0.0
    >>> LA.norm(b, -np.inf)
    2.0
    >>> LA.norm(a, 1)
    20.0
    >>> LA.norm(b, 1)
    7.0
    >>> LA.norm(a, -1)
    -4.6566128774142013e-010
    >>> LA.norm(b, -1)
    6.0
    >>> LA.norm(a, 2)
    7.745966692414834
    >>> LA.norm(b, 2)
    7.3484692283495345
    >>> LA.norm(a, -2)
    0.0
    >>> LA.norm(b, -2)
    1.8570331885190563e-016 # may vary
    >>> LA.norm(a, 3)
    5.8480354764257312 # may vary
    >>> LA.norm(a, -3)
    0.0
    Using the `axis` argument to compute vector norms:
    >>> c = np.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1)
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1)
    array([ 6.,  6.])
    Using the `axis` argument to compute matrix norms:
    >>> m = np.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (3.7416573867739413, 11.224972160321824)
    """
    if axis is not None and hasattr(axis, "__len__") and len(axis) > 2:
        return T.power(
            T.power(T.abs(x), ord).sum(axis=axis, keepdims=keepdims), 1.0 / ord
        )
    else:
        return _norm(x, ord, axis, keepdims)


def singular_vectors_power_iteration(weight, axis=0, n_iters=1):

    # This power iteration produces approximations of `u` and `v`.

    u = normalize(random.randn(weight.shape[0]), dim=0)
    v = normalize(random.randn(weight.shape[1]), dim=0)

    for _ in range(n_iters):

        v = normalize(weight.t().dot(u), dim=0)
        u = normalize(weight.dot(v), dim=0)

    return u, v


def eigenvector_power_iteration(weight, axis=0, n_iters=1):

    # This power iteration produces approximations of `u`.

    u = normalize(random.randn(weight.shape[0]), dim=0)

    for _ in range(n_iters):

        u = normalize(weight.t().dot(u), dim=0)

    return u
