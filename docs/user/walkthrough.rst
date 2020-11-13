.. _walkthrough:

Computational Graph
===================

In SymJAX, calling an operation like `a=T.ones((10,))` does not perform any acutal computation of the 10 dimensional vector. Instead, SymJAX creates a node that only knows about the shape, the type, the internal operation to apply, and the parents of the operation. In this case for example, `a` is an `Op` with internal function `jax.numpy.ones`, with shape being `(10,)` and dtype being `float64` by default. The parents of this node are the function inputs, in this case the tuple `(10,)`.
As a result, when defining a computation graph, one simply creates multiple nodes that are interconnected, the whole forming an directed acyclic graph (DAG).
Only when creating a compiled function with `symjax.function` or when performing a lazy evaluation with `.get` the graph will be traversed and possibly evaluated in the latter case. In both scenarios, one need to traverse the graph, we describe our procedure below.

Graph traversing
================


Graph traversing can be come in a straightforward manner in a recursive manner. One simply starts from a node given by the user, find the node parents and recurse this function until the found node can be given an expicit value (if it is a `T.Variable`, a `T.Placeholder`, or a `T.Constant` node). This is done in a bottom to top fashion. From that case the explicit value can be passed to the children nodes jax function to evaluate each node in a top to bottom fashion.

Recursion has a few limitations. The main one being the recursion stack limit when allows to use such paradigm only on small graph to avoid memory issues. Performance is also often hindered in Python by recursion as opposed to a direct implementation due to the stack memory management (this may vary based on the case at hand).
We thus opted for a non-recursive approach.