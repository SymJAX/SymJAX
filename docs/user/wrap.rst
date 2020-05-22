Wrap Jax computation into SymJAX
================================

The computation in Jax is done eagerly similarly to TF2 and PyTorch. In SymJAX
the computational graph definition is done a priori with symbolic variables.
That is, no actual computations are done during the graph definition, once done
the graph is compiled with proper inputs/outputs/updates to provide the user
with a compiled function executing the graph. This graph thus involves various
operations, one can define its own in the two following way. First by combining
the already existing SymJAX function, the other by creating it in pure Jax and
then wrapping it into a SymJAX symbolic operation as demonstrated below.

.. literalinclude:: ../../examples/wrap.py

A SymJAX computation graph can not be partially defined with Jax computation,
the above thus provides an easy way to wrap Jax computations into a SymJAX Op
which can then be put into the graph as any other SymJAX provided Ops.

