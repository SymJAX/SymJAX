 .. _symjax-tensor:

:mod:`symjax.tensor`
--------------------


.. automodule:: symjax.tensor

Implements the NumPy API, using the primitives in :mod:`jax.lax`.
As SymJAX follows the JAX restrictions, not all NumPy functins are present.

* Notably, since JAX arrays are immutable, NumPy APIs that mutate arrays
  in-place cannot be implemented in JAX. However, often JAX is able to provide a
  alternative API that is purely functional. For example, instead of in-place
  array updates (:code:`x[i] = y`), JAX provides an alternative pure indexed
  update function :func:`jax.ops.index_update`.

* NumPy is very aggressive at promoting values to :code:`float64` type. JAX
  sometimes is less aggressive about type promotion.

Finally, since SymJAX uses jit-compilation, any function that returns
data-dependent output shapes are incompatible and thus not implemented.
In fact, The XLA compiler requires that shapes of arrays be known at
compile time. While it would be possible to provide. Thus an implementation of an API such as :func:`numpy.nonzero`, we would be unable
to JIT-compile it because the shape of its output depends on the contents
of the input data.

Not every function in NumPy is implemented; contributions are welcome!


Numpy Ops
=========


.. autosummary::

    abs
    absolute
    add
    all
    allclose
    alltrue
    amax
    amin
    angle
    any
    append
    arange
    arccos
    arccosh
    arcsin
    arcsinh
    arctan
    arctan2
    arctanh
    argmax
    argmin
    argsort
    argwhere
    around
    array
    array_repr
    array_str
    asarray
    atleast_1d
    atleast_2d
    atleast_3d
    bartlett
    bincount
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    blackman
    block
    broadcast_arrays
    broadcast_to
    can_cast
    ceil
    clip
    column_stack
    compress
    concatenate
    conj
    conjugate
    convolve
    copysign
    corrcoef
    correlate
    cos
    cosh
    count_nonzero
    cov
    cross
    cumsum
    cumprod
    cumproduct
    deg2rad
    degrees
    diag
    diag_indices
    diag_indices_from
    diagflat
    diagonal
    digitize
    divide
    divmod
    dot
    dsplit
    dstack
    ediff1d
    einsum
    equal
    empty
    empty_like
    exp
    exp2
    expand_dims
    expm1
    extract
    eye
    fabs
    fix
    flatnonzero
    flip
    fliplr
    flipud
    float_power
    floor
    floor_divide
    fmax
    fmin
    fmod
    frexp
    full
    full_like
    gcd
    geomspace
    greater
    greater_equal
    hamming
    hanning
    heaviside
    histogram
    histogram_bin_edges
    hsplit
    hstack
    hypot
    identity
    imag
    in1d
    indices
    inner
    isclose
    iscomplex
    isfinite
    isin
    isinf
    isnan
    isneginf
    isposinf
    isreal
    isscalar
    issubdtype
    issubsctype
    ix_
    kaiser
    kron
    lcm
    ldexp
    left_shift
    less
    less_equal
    linspace
    log
    log10
    log1p
    log2
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logspace
    matmul
    max
    maximum
    mean
    median
    meshgrid
    min
    minimum
    mod
    moveaxis
    msort
    multiply
    nan_to_num
    nanargmax
    nanargmin
    nancumprod
    nancumsum
    nanmax
    nanmedian
    nanmin
    nanpercentile
    nanprod
    nanquantile
    nansum
    negative
    nextafter
    nonzero
    not_equal
    ones
    ones_like
    outer
    packbits
    pad
    percentile
    polyadd
    polyder
    polymul
    polysub
    polyval
    power
    positive
    prod
    product
    promote_types
    ptp
    quantile
    rad2deg
    radians
    ravel
    real
    reciprocal
    remainder
    repeat
    reshape
    result_type
    right_shift
    rint
    roll
    rollaxis
    roots
    rot90
    round
    row_stack
    searchsorted
    select
    sign
    signbit
    sin
    sinc
    sinh
    sometrue
    sort
    split
    sqrt
    square
    squeeze
    stack
    std
    subtract
    sum
    swapaxes
    take
    take_along_axis
    tan
    tanh
    tensordot
    tile
    trace
    transpose
    tri
    tril
    tril_indices
    tril_indices_from
    triu
    triu_indices
    triu_indices_from
    true_divide
    trunc
    unique
    unpackbits
    unravel_index
    unwrap
    vander
    var
    vdot
    vsplit
    vstack
    where
    zeros
    zeros_like
    stop_gradient
    one_hot
    dimshuffle
    flatten
    flatten2d
    flatten3d
    flatten4d

Indexed Operations
==================

.. autosummary::

    index
    index_update
    index_min
    index_add
    index_max
    index_take
    index_in_dim
    dynamic_slice_in_dim
    dynamic_slice
    dynamic_index_in_dim



Control flow Ops
================

.. autosummary::

    cond
    fori_loop
    map
    scan
    while_loop





Detailed Descriptions
=====================


.. autofunction:: abs
.. autofunction:: absolute
.. autofunction:: add
.. autofunction:: all
.. autofunction:: allclose
.. autofunction:: alltrue
.. autofunction:: amax
.. autofunction:: amin
.. autofunction:: angle
.. autofunction:: any
.. autofunction:: append
.. autofunction:: arange
.. autofunction:: arccos
.. autofunction:: arccosh
.. autofunction:: arcsin
.. autofunction:: arcsinh
.. autofunction:: arctan
.. autofunction:: arctan2
.. autofunction:: arctanh
.. autofunction:: argmax
.. autofunction:: argmin
.. autofunction:: argsort
.. autofunction:: around
.. autofunction:: asarray
.. autofunction:: atleast_1d
.. autofunction:: atleast_2d
.. autofunction:: atleast_3d
.. autofunction:: bitwise_and
.. autofunction:: bitwise_not
.. autofunction:: bitwise_or
.. autofunction:: bitwise_xor
.. autofunction:: block
.. autofunction:: broadcast_arrays
.. autofunction:: broadcast_to
.. autofunction:: can_cast
.. autofunction:: ceil
.. autofunction:: clip
.. autofunction:: column_stack
.. autofunction:: concatenate
.. autofunction:: conj
.. autofunction:: conjugate
.. autofunction:: corrcoef
.. autofunction:: cos
.. autofunction:: cosh
.. autofunction:: count_nonzero
.. autofunction:: cov
.. autofunction:: cross
.. autofunction:: cumsum
.. autofunction:: cumprod
.. autofunction:: cumproduct
.. autofunction:: deg2rad
.. autofunction:: degrees
.. autofunction:: diag
.. autofunction:: diag_indices
.. autofunction:: diagonal
.. autofunction:: divide
.. autofunction:: divmod
.. autofunction:: dot
.. autofunction:: dsplit
.. autofunction:: dstack
.. autofunction:: einsum
.. autofunction:: equal
.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: exp
.. autofunction:: exp2
.. autofunction:: expand_dims
.. autofunction:: expm1
.. autofunction:: eye
.. autofunction:: fabs
.. autofunction:: fix
.. autofunction:: flip
.. autofunction:: fliplr
.. autofunction:: flipud
.. autofunction:: float_power
.. autofunction:: floor
.. autofunction:: floor_divide
.. autofunction:: fmod
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: gcd
.. autofunction:: geomspace
.. autofunction:: greater
.. autofunction:: greater_equal
.. autofunction:: heaviside
.. autofunction:: hsplit
.. autofunction:: hstack
.. autofunction:: hypot
.. autofunction:: identity
.. autofunction:: imag
.. autofunction:: inner
.. autofunction:: isclose
.. autofunction:: iscomplex
.. autofunction:: isfinite
.. autofunction:: isinf
.. autofunction:: isnan
.. autofunction:: isneginf
.. autofunction:: isposinf
.. autofunction:: isreal
.. autofunction:: isscalar
.. autofunction:: issubdtype
.. autofunction:: issubsctype
.. autofunction:: ix_
.. autofunction:: kron
.. autofunction:: lcm
.. autofunction:: left_shift
.. autofunction:: less
.. autofunction:: less_equal
.. autofunction:: linspace
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: log1p
.. autofunction:: log2
.. autofunction:: logaddexp
.. autofunction:: logaddexp2
.. autofunction:: logical_and
.. autofunction:: logical_not
.. autofunction:: logical_or
.. autofunction:: logical_xor
.. autofunction:: logspace
.. autofunction:: matmul
.. autofunction:: max
.. autofunction:: maximum
.. autofunction:: mean
.. autofunction:: median
.. autofunction:: meshgrid
.. autofunction:: min
.. autofunction:: minimum
.. autofunction:: mod
.. autofunction:: moveaxis
.. autofunction:: multiply
.. autofunction:: nan_to_num
.. autofunction:: nancumprod
.. autofunction:: nancumsum
.. autofunction:: nanmax
.. autofunction:: nanmin
.. autofunction:: nanprod
.. autofunction:: nansum
.. autofunction:: negative
.. autofunction:: nextafter
.. autofunction:: nonzero
.. autofunction:: not_equal
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: outer
.. autofunction:: pad
.. autofunction:: percentile
.. autofunction:: polyval
.. autofunction:: power
.. autofunction:: positive
.. autofunction:: prod
.. autofunction:: product
.. autofunction:: promote_types
.. autofunction:: ptp
.. autofunction:: quantile
.. autofunction:: rad2deg
.. autofunction:: radians
.. autofunction:: ravel
.. autofunction:: real
.. autofunction:: reciprocal
.. autofunction:: remainder
.. autofunction:: repeat
.. autofunction:: reshape
.. autofunction:: result_type
.. autofunction:: right_shift
.. autofunction:: roll
.. autofunction:: rot90
.. autofunction:: round
.. autofunction:: row_stack
.. autofunction:: select
.. autofunction:: sign
.. autofunction:: signbit
.. autofunction:: sin
.. autofunction:: sinc
.. autofunction:: sinh
.. autofunction:: sometrue
.. autofunction:: sort
.. autofunction:: split
.. autofunction:: sqrt
.. autofunction:: square
.. autofunction:: squeeze
.. autofunction:: stack
.. autofunction:: std
.. autofunction:: subtract
.. autofunction:: sum
.. autofunction:: swapaxes
.. autofunction:: take
.. autofunction:: take_along_axis
.. autofunction:: tan
.. autofunction:: tanh
.. autofunction:: tensordot
.. autofunction:: tile
.. autofunction:: trace
.. autofunction:: transpose
.. autofunction:: tri
.. autofunction:: tril
.. autofunction:: tril_indices
.. autofunction:: triu
.. autofunction:: triu_indices
.. autofunction:: true_divide
.. autofunction:: vander
.. autofunction:: var
.. autofunction:: vdot
.. autofunction:: vsplit
.. autofunction:: vstack
.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: stop_gradient
.. autofunction:: one_hot
.. autofunction:: dimshuffle
.. autofunction:: flatten
.. autofunction:: flatten2d
.. autofunction:: flatten3d
.. autofunction:: flatten4d


.. autofunction:: index
.. autofunction:: index_update
.. autofunction:: index_min
.. autofunction:: index_add
.. autofunction:: index_max
.. autofunction:: index_take
.. autofunction:: index_in_dim
.. autofunction:: dynamic_slice_in_dim
.. autofunction:: dynamic_slice
.. autofunction:: dynamic_index_in_dim


.. automodule:: control_flow
   :members:



Extra
=====

.. automodule:: ops_special
   :members:


