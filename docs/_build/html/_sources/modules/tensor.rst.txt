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

For/While/Scan
==============

.. automodule:: symjax.tensor.control_flow
   :members:

Numpy
=====


.. autofunction:: symjax.tensor.abs
.. autofunction:: symjax.tensor.absolute
.. autofunction:: symjax.tensor.add
.. autofunction:: symjax.tensor.all
.. autofunction:: symjax.tensor.allclose
.. autofunction:: symjax.tensor.alltrue
.. autofunction:: symjax.tensor.amax
.. autofunction:: symjax.tensor.amin
.. autofunction:: symjax.tensor.angle
.. autofunction:: symjax.tensor.any
.. autofunction:: symjax.tensor.append
.. autofunction:: symjax.tensor.arange
.. autofunction:: symjax.tensor.arccos
.. autofunction:: symjax.tensor.arccosh
.. autofunction:: symjax.tensor.arcsin
.. autofunction:: symjax.tensor.arcsinh
.. autofunction:: symjax.tensor.arctan
.. autofunction:: symjax.tensor.arctan2
.. autofunction:: symjax.tensor.arctanh
.. autofunction:: symjax.tensor.argmax
.. autofunction:: symjax.tensor.argmin
.. autofunction:: symjax.tensor.argsort
.. autofunction:: symjax.tensor.around
.. autofunction:: symjax.tensor.asarray
.. autofunction:: symjax.tensor.atleast_1d
.. autofunction:: symjax.tensor.atleast_2d
.. autofunction:: symjax.tensor.atleast_3d
.. autofunction:: symjax.tensor.bitwise_and
.. autofunction:: symjax.tensor.bitwise_not
.. autofunction:: symjax.tensor.bitwise_or
.. autofunction:: symjax.tensor.bitwise_xor
.. autofunction:: symjax.tensor.block
.. autofunction:: symjax.tensor.broadcast_arrays
.. autofunction:: symjax.tensor.broadcast_to
.. autofunction:: symjax.tensor.can_cast
.. autofunction:: symjax.tensor.ceil
.. autofunction:: symjax.tensor.clip
.. autofunction:: symjax.tensor.column_stack
.. autofunction:: symjax.tensor.concatenate
.. autofunction:: symjax.tensor.conj
.. autofunction:: symjax.tensor.conjugate
.. autofunction:: symjax.tensor.corrcoef
.. autofunction:: symjax.tensor.cos
.. autofunction:: symjax.tensor.cosh
.. autofunction:: symjax.tensor.count_nonzero
.. autofunction:: symjax.tensor.cov
.. autofunction:: symjax.tensor.cross
.. autofunction:: symjax.tensor.cumsum
.. autofunction:: symjax.tensor.cumprod
.. autofunction:: symjax.tensor.cumproduct
.. autofunction:: symjax.tensor.deg2rad
.. autofunction:: symjax.tensor.degrees
.. autofunction:: symjax.tensor.diag
.. autofunction:: symjax.tensor.diag_indices
.. autofunction:: symjax.tensor.diagonal
.. autofunction:: symjax.tensor.divide
.. autofunction:: symjax.tensor.divmod
.. autofunction:: symjax.tensor.dot
.. autofunction:: symjax.tensor.dsplit
.. autofunction:: symjax.tensor.dstack
.. autofunction:: symjax.tensor.einsum
.. autofunction:: symjax.tensor.equal
.. autofunction:: symjax.tensor.empty
.. autofunction:: symjax.tensor.empty_like
.. autofunction:: symjax.tensor.exp
.. autofunction:: symjax.tensor.exp2
.. autofunction:: symjax.tensor.expand_dims
.. autofunction:: symjax.tensor.expm1
.. autofunction:: symjax.tensor.eye
.. autofunction:: symjax.tensor.fabs
.. autofunction:: symjax.tensor.fix
.. autofunction:: symjax.tensor.flip
.. autofunction:: symjax.tensor.fliplr
.. autofunction:: symjax.tensor.flipud
.. autofunction:: symjax.tensor.float_power
.. autofunction:: symjax.tensor.floor
.. autofunction:: symjax.tensor.floor_divide
.. autofunction:: symjax.tensor.fmod
.. autofunction:: symjax.tensor.full
.. autofunction:: symjax.tensor.full_like
.. autofunction:: symjax.tensor.gcd
.. autofunction:: symjax.tensor.geomspace
.. autofunction:: symjax.tensor.greater
.. autofunction:: symjax.tensor.greater_equal
.. autofunction:: symjax.tensor.heaviside
.. autofunction:: symjax.tensor.hsplit
.. autofunction:: symjax.tensor.hstack
.. autofunction:: symjax.tensor.hypot
.. autofunction:: symjax.tensor.identity
.. autofunction:: symjax.tensor.imag
.. autofunction:: symjax.tensor.inner
.. autofunction:: symjax.tensor.isclose
.. autofunction:: symjax.tensor.iscomplex
.. autofunction:: symjax.tensor.isfinite
.. autofunction:: symjax.tensor.isinf
.. autofunction:: symjax.tensor.isnan
.. autofunction:: symjax.tensor.isneginf
.. autofunction:: symjax.tensor.isposinf
.. autofunction:: symjax.tensor.isreal
.. autofunction:: symjax.tensor.isscalar
.. autofunction:: symjax.tensor.issubdtype
.. autofunction:: symjax.tensor.issubsctype
.. autofunction:: symjax.tensor.ix_
.. autofunction:: symjax.tensor.kron
.. autofunction:: symjax.tensor.lcm
.. autofunction:: symjax.tensor.left_shift
.. autofunction:: symjax.tensor.less
.. autofunction:: symjax.tensor.less_equal
.. autofunction:: symjax.tensor.linspace
.. autofunction:: symjax.tensor.log
.. autofunction:: symjax.tensor.log10
.. autofunction:: symjax.tensor.log1p
.. autofunction:: symjax.tensor.log2
.. autofunction:: symjax.tensor.logaddexp
.. autofunction:: symjax.tensor.logaddexp2
.. autofunction:: symjax.tensor.logical_and
.. autofunction:: symjax.tensor.logical_not
.. autofunction:: symjax.tensor.logical_or
.. autofunction:: symjax.tensor.logical_xor
.. autofunction:: symjax.tensor.logspace
.. autofunction:: symjax.tensor.matmul
.. autofunction:: symjax.tensor.max
.. autofunction:: symjax.tensor.maximum
.. autofunction:: symjax.tensor.mean
.. autofunction:: symjax.tensor.median
.. autofunction:: symjax.tensor.meshgrid
.. autofunction:: symjax.tensor.min
.. autofunction:: symjax.tensor.minimum
.. autofunction:: symjax.tensor.mod
.. autofunction:: symjax.tensor.moveaxis
.. autofunction:: symjax.tensor.multiply
.. autofunction:: symjax.tensor.nan_to_num
.. autofunction:: symjax.tensor.nancumprod
.. autofunction:: symjax.tensor.nancumsum
.. autofunction:: symjax.tensor.nanmax
.. autofunction:: symjax.tensor.nanmin
.. autofunction:: symjax.tensor.nanprod
.. autofunction:: symjax.tensor.nansum
.. autofunction:: symjax.tensor.negative
.. autofunction:: symjax.tensor.nextafter
.. autofunction:: symjax.tensor.nonzero
.. autofunction:: symjax.tensor.not_equal
.. autofunction:: symjax.tensor.ones
.. autofunction:: symjax.tensor.ones_like
.. autofunction:: symjax.tensor.outer
.. autofunction:: symjax.tensor.pad
.. autofunction:: symjax.tensor.percentile
.. autofunction:: symjax.tensor.polyval
.. autofunction:: symjax.tensor.power
.. autofunction:: symjax.tensor.positive
.. autofunction:: symjax.tensor.prod
.. autofunction:: symjax.tensor.product
.. autofunction:: symjax.tensor.promote_types
.. autofunction:: symjax.tensor.ptp
.. autofunction:: symjax.tensor.quantile
.. autofunction:: symjax.tensor.rad2deg
.. autofunction:: symjax.tensor.radians
.. autofunction:: symjax.tensor.ravel
.. autofunction:: symjax.tensor.real
.. autofunction:: symjax.tensor.reciprocal
.. autofunction:: symjax.tensor.remainder
.. autofunction:: symjax.tensor.repeat
.. autofunction:: symjax.tensor.reshape
.. autofunction:: symjax.tensor.result_type
.. autofunction:: symjax.tensor.right_shift
.. autofunction:: symjax.tensor.roll
.. autofunction:: symjax.tensor.rot90
.. autofunction:: symjax.tensor.round
.. autofunction:: symjax.tensor.row_stack
.. autofunction:: symjax.tensor.select
.. autofunction:: symjax.tensor.sign
.. autofunction:: symjax.tensor.signbit
.. autofunction:: symjax.tensor.sin
.. autofunction:: symjax.tensor.sinc
.. autofunction:: symjax.tensor.sinh
.. autofunction:: symjax.tensor.sometrue
.. autofunction:: symjax.tensor.sort
.. autofunction:: symjax.tensor.split
.. autofunction:: symjax.tensor.sqrt
.. autofunction:: symjax.tensor.square
.. autofunction:: symjax.tensor.squeeze
.. autofunction:: symjax.tensor.stack
.. autofunction:: symjax.tensor.std
.. autofunction:: symjax.tensor.subtract
.. autofunction:: symjax.tensor.sum
.. autofunction:: symjax.tensor.swapaxes
.. autofunction:: symjax.tensor.take
.. autofunction:: symjax.tensor.take_along_axis
.. autofunction:: symjax.tensor.tan
.. autofunction:: symjax.tensor.tanh
.. autofunction:: symjax.tensor.tensordot
.. autofunction:: symjax.tensor.tile
.. autofunction:: symjax.tensor.trace
.. autofunction:: symjax.tensor.transpose
.. autofunction:: symjax.tensor.tri
.. autofunction:: symjax.tensor.tril
.. autofunction:: symjax.tensor.tril_indices
.. autofunction:: symjax.tensor.triu
.. autofunction:: symjax.tensor.triu_indices
.. autofunction:: symjax.tensor.true_divide
.. autofunction:: symjax.tensor.vander
.. autofunction:: symjax.tensor.var
.. autofunction:: symjax.tensor.vdot
.. autofunction:: symjax.tensor.vsplit
.. autofunction:: symjax.tensor.vstack
.. autofunction:: symjax.tensor.zeros
.. autofunction:: symjax.tensor.zeros_like

Extra
=====

.. automodule:: symjax.tensor.ops_special
   :members:


