import jax.lax as jla

from .base import jax_wrap, symjax_to_jax_fn

cond = jax_wrap(jla.cond)
fori_loop = jax_wrap(jla.fori_loop)
while_loop = jax_wrap(jla.while_loop)


@jax_wrap
def _scan(f, init, sequences, non_sequences=None, length=None, reverse=False):
    # get the fully jaxed function
    truef = symjax_to_jax_fn(f)

    # now create a dummy function that only takes as input the sequences
    if non_sequences is None:

        def finalf(a, args):
            return truef(a, *args)

    else:

        def finalf(a, args):
            return truef(a, *args, *non_sequences)

    return jla.scan(finalf, init, sequences, length=length, reverse=reverse)


@jax_wrap
def _while_loop(
    cond_fun, body_fun, sequences, non_sequences_cond=None, non_sequences_body=None
):

    # get the fully jaxed function
    truecond = symjax_to_jax_fn(cond_fun)
    truebody = symjax_to_jax_fn(body_fun)

    # now create a dummy function that only takes as input the sequences
    if non_sequences_cond is None:
        finalcond = truecond
    else:

        def finalcond(args):
            return truecond(args, *non_sequences_cond)

    if non_sequences_body is None:
        finalbody = truebody
    else:

        def finalbody(args):
            return truebody(args, *non_sequences_body)

    return jla.while_loop(finalcond, finalbody, sequences)


def map(f, sequences, non_sequences=None):
    """Map a function over leading array axes.

    Like Python's builtin map, except inputs and outputs are in the form of
    stacked arrays. Consider using the ``jax.vmap`` transform instead, unless you
    need to apply a function element by element for reduced memory usage or
    heterogeneous computation with other control flow primitives.

    When ``xs`` is an array type, the semantics of ``map`` are given by this
    Python implementation::

      def map(f, xs):
        return np.stack([f(x) for x in xs])

    Like ``scan``, ``map`` is implemented in terms of JAX primitives so many of
    the same advantages over a Python loop apply: ``xs`` may be an arbitrary
    nested pytree type, and the mapped computation is compiled only once.

    Args:
      f: a Python function to apply element-wise over the first axis or axes of
        ``sequences``.
      sequences: list of values over which to map along the leading axis.
      non_sequences: list of values passed the same at each call

    Returns:
      Mapped values.

    Example:
        example of creating a diagonal matrix:
        .. doctest::

           >>> import symjax.tensor as T
           >>> import symjax
           >>> x = T.ones(3)
           >>> y = T.zeros(3)
           >>> w = T.arange(3)
           >>> out = T.map(lambda x, i, w: T.index_update(w, i, x), sequences=[x, w], non_sequences=[y])
           >>> f = symjax.function(outputs=out)
           >>> f()
           array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=float32)
    """

    g = lambda _, *args: (1, f(*args))

    if type(non_sequences) == list:
        non_sequences = tuple(non_sequences)
    if type(sequences) == list:
        sequences = tuple(sequences)
    ys = scan(g, 0, sequences, non_sequences=non_sequences)[1]

    return ys


def scan(f, init, sequences, non_sequences=None, length=None, reverse=False):
    """Scan a function over leading array axes while carrying along state.

    The type signature in brief is

    .. code-block:: haskell

      scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

    where we use [t] here to denote the type t with an additional leading axis.
    That is, if t is an array type then [t] represents the type with an additional
    leading axis, and if t is a pytree (container) type with array leaves then [t]
    represents the type with the same pytree structure and corresponding leaves
    each with an additional leading axis.

    When ``a`` is an array type or None, and ``b`` is an array type, the semantics
    of ``scan`` are given roughly by this Python implementation::

      def scan(f, init, xs, length=None):
        if xs is None:
          xs = [None] * length
        carry = init
        ys = []
        for x in xs:
          carry, y = f(carry, x)
          ys.append(y)
        return carry, np.stack(ys)

    Unlike that Python version, both ``a`` and ``b`` may be arbitrary pytree
    types, and so multiple arrays can be scanned over at once and produce multiple
    output arrays. (None is actually an empty pytree.)

    Also unlike that Python version, ``scan`` is a JAX primitive and is lowered to
    a single XLA While HLO. That makes it useful for reducing compilation times
    for jit-compiled functions, since native Python loop constructs in an ``@jit``
    function are unrolled, leading to large XLA computations.

    Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
    across all iterations (and not just be consistent up to NumPy rank/shape
    broadcasting and dtype promotion rules, for example). In other words, the type
    ``c`` in the type signature above represents an array with a fixed shape and
    dtype (or a nested tuple/list/dict container data structure with a fixed
    structure and arrays with fixed shape and dtype at the leaves).

    Args:
      f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
        that ``f`` accepts two arguments where the first is a value of the loop
        carry and the second is a slice of ``xs`` along its leading axis, and that
        ``f`` returns a pair where the first element represents a new value for
        the loop carry and the second represents a slice of the output.
      init: an initial loop carry value of type ``c``, which can be a scalar,
        array, or any pytree (nested Python tuple/list/dict) thereof, representing
        the initial loop carry value. This value must have the same structure as
        the first element of the pair returned by ``f``.
      sequences: list of values over which to scan along the leading axis,
        where ``[a]`` can be an array or any pytree (nested Python
        tuple/list/dict) thereof with consistent leading axis sizes.
      non_sequences: (optional) list of values that do not depend of the
        iterations but are used in the function
      length: optional integer specifying the number of loop iterations, which
        must agree with the sizes of leading axes of the arrays in ``sequences`` (but can
        be used to perform scans where no input ``xs`` are needed).
      reverse: optional boolean specifying whether to run the scan iteration
        forward (the default) or in reverse, equivalent to reversing the leading
        axes of the arrays in both ``xs`` and in ``ys``.

    Returns:
      A pair of type ``(c, [b])`` where the first element represents the final
      loop carry value and the second element represents the stacked outputs of
      the second output of ``f`` when scanned over the leading axis of the inputs.

    Example:
        .. doctest::

           >>> import symjax.tensor as T
           >>> import symjax
           >>> indices = T.Placeholder((3,), 'int32')
           >>> w = T.arange(10)
           >>> out = T.scan(lambda acc, i, w, : (acc + w[i],i), T.zeros(1), sequences=[indices], non_sequences=[w])
           >>> # out first element is the carry, the second is the stacked second outputs of the function
           >>> f = symjax.function(indices, outputs=out)
           >>> f([0,2,3])
           [array([5.], dtype=float32), array([0, 2, 3], dtype=int32)]
           >>> f([1, 1, 1])
           [array([3.], dtype=float32), array([1, 1, 1], dtype=int32)]
    """
    if type(non_sequences) == list:
        non_sequences = tuple(non_sequences)

    if type(sequences) == list:
        sequences = tuple(sequences)

    return _scan(f, init, sequences, non_sequences, length, reverse)


def while_loop(
    cond_fun, body_fun, sequences, non_sequences_cond=None, non_sequences_body=None
):
    """Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

     The type signature in brief is

     .. code-block:: haskell

       while_loop :: (a -> Bool) -> (a -> a) -> a -> a

     The semantics of ``while_loop`` are given by this Python implementation::

       def while_loop(cond_fun, body_fun, init_val):
         val = init_val
         while cond_fun(val):
           val = body_fun(val)
         return val

     Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
     to a single XLA While HLO. That makes it useful for reducing compilation times
     for jit-compiled functions, since native Python loop constructs in an ``@jit``
     function are unrolled, leading to large XLA computations.

     Also unlike the Python analogue, the loop-carried value ``val`` must hold a
     fixed shape and dtype across all iterations (and not just be consistent up to
     NumPy rank/shape broadcasting and dtype promotion rules, for example). In
     other words, the type ``a`` in the type signature above represents an array
     with a fixed shape and dtype (or a nested tuple/list/dict container data
     structure with a fixed structure and arrays with fixed shape and dtype at the
     leaves).

     Another difference from using Python-native loop constructs is that
     ``while_loop`` is not reverse-mode differentiable because XLA computations
     require static bounds on memory requirements.

     Args:
       cond_fun: function of type ``a -> Bool``.
       body_fun: function of type ``a -> a``.
       init_val: value of type ``a``, a type that can be a scalar, array, or any
         pytree (nested Python tuple/list/dict) thereof, representing the initial
         loop carry value.

     Returns:
       The output from the final iteration of body_fun, of type ``a``.
    """

    if type(non_sequences_cond) == list:
        non_sequences_cond = tuple(non_sequences_cond)
    if type(non_sequences_body) == list:
        non_sequences_body = tuple(non_sequences_body)
    if type(sequences) == list:
        sequences = tuple(sequences)

    return _while_loop(
        cond_fun, body_fun, sequences, non_sequences_cond, non_sequences_body
    )
