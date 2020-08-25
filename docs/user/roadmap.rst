.. _roadmap:

Roadmap
=======

Brief description of the incoming SymJAX updates.

Short-Term
----------

- graph saving and loading (right now one can only save and load variables, this
  requires the user to redefine a graph prior loading the saved variables in order
  to keep training/updating/evaluating a graph, saving/loading the graph itself
  would remove this need
- function based data/output saving in h5 files allowing to (in real time)
  save and monitor specified quantities evaluated through each function


Long-Term
---------

- support of the ``vmap`` and ``pmap`` Jax functions allowing distributed and
  optimized computations for more general graphs
- general wrapping of Tensorflow modules using Jax as backend
