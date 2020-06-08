Quick Walkthrough Tutorial of SymJAX
====================================

.. toctree::
  :hidden:

We briefly describe some key components of SymJAX.


Function: compiling a graph into an executable (function)
---------------------------------------------------------

As opposed to most current softwares, SymJAX proposes a symbolic viewpoint
from which one can create a computational graph, laying out all the computation
pipeline from inputs to outputs including updates of persistent variables. Once
this is defined, it is possible to compile this graph to optimize the exection
speed. In fact, knowing the graph (nodes, connections, shapes, types, constant
values) is enough to produce an highly optimized executable of this graph. In
SymJAX this is done via  symjax.function as demonstrated below:

.. literalinclude:: ../../examples/function.py


Clone: one line multipurpose graph replacement
----------------------------------------------

In most current packages, the ability to perform an already define computation
graph but with altered nodes is cumbersone. Some specific involve the use
of layers as in Keras where one can feed any value hence allow to compute a
feedforward pass without much changes but if one had to replace a specific
variable or more complex part of a graph no tools are available. In Theano,
the clone function allowed to do such thing and it implemented in SymJAX as
well. As per the below example, it is clear how the clone utility allows to
get an already defined computational graph and replace any subgraph in it with
another based on a node->node mapping:

.. literalinclude:: ../../examples/clone.py

Scopes, Operations/Variables/Placeholders naming and accessing
--------------------------------------------------------------

Accessing, naming variables, operations and placeholders. This is done in a
similar way as in the vanilla Tensorflow form with scopes and EVERY of the
variable/placeholder/operation is named and located with a unique identifier
(name) per scope. If during creation both have same names, the original name is
augmented with an underscore and interger number, here is a brief example:

.. literalinclude:: ../../examples/graph_accessing.py

Graph Saving and Loading
------------------------

An important feature of SymJAX is the easiness to reset, save, load variables.
This is crucial in order to save a model and being to reloaded (in a possibly
different script) to keep using it. In our case, a computational graph is
completely defined by its structure and the values of the persistent nodes 
(the variables). Hence, it is enough to save the variables. This is done in a
very explicit manner using the numpy.savez utility where the saved file can
be accessed from any other script, variables can be loaded, accessed, even
modified, and then reloaded inside the computational graph. Here is a brief
example:

.. literalinclude:: ../../examples/graph.py


Wrap: Jax function/computation to SymJAX Op
-------------------------------------------

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

