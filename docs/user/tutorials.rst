Tutorials
=========

SymJAX
''''''

We briefly describe some key components of SymJAX.


.. _function:

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


While/Map/Scan
--------------

An important part of many implementations resides in the use of for/while loops
and in scans, which allow to maintain and update an additional quantity through
the iterations. In SymJAX, those operators are different from the Jax ones and
closers to the Theano ones as they provide an explicit ``sequences`` and
``non_sequences`` argument. Here are a few examples below:

.. literalinclude:: ../../examples/control_flow.py

the use of the ``non_sequences`` argument allows to keep track of the internal
function dependencies without requiring to execute the function. Hence all
tensors used inside a function should be part of the ``sequences`` or
``non_sequences`` op inputs.

.. _none:

Variable batch length (shape)
-----------------------------

In many applications it is required to have length varying inputs to a compiled
SymJAX function. This can be done by expliciting setting the shape of the corresponding
``Placeholders`` to 0 (this will likely change in the future) as demonstrated
below:

.. literalinclude:: ../../examples/vmap.py

in the backend, SymJAX automatically jit the overall (vmapped) functions for
optimal performances.



.. _viz:

Graph visualization
-------------------

Similarly to Theano, it is possible to display the computational graph of the code written as follows:


.. literalinclude:: ../../examples/viz_graph.py


.. image:: ../img/file.png


.. _clone:

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


.. _scopes:

Scopes, Operations/Variables/Placeholders naming and accessing
--------------------------------------------------------------

Accessing, naming variables, operations and placeholders. This is done in a
similar way as in the vanilla Tensorflow form with scopes and EVERY of the
variable/placeholder/operation is named and located with a unique identifier
(name) per scope. If during creation both have same names, the original name is
augmented with an underscore and interger number, here is a brief example:

.. literalinclude:: ../../examples/graph_accessing.py


.. _saving:

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


.. _wrapf:

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


.. _wrapc:

Wrap: Jax class to SymJAX class
-------------------------------

One might have defined a Jax class, with a constructor possibly taking some 
constant values and some jax arrays, performing some computations, setting
some attributes, and then interacting with those attributes when calling the
class methods. It would be particularly easy to pair such already implemented
classes with SymJAX computation graph. This can be done as follows:

.. literalinclude:: ../../examples/wrap_class.py

As can be seen, there is some restrictions. First, the behavior inside the
constructor of the original class should be fixed as it will be executed once
by the wrapper in order to map the constructor computations into SymJAX. 
Second, any jax array update done internally will break the conversion as
such operations are only allowed for Variables in SymJAX, hence some care is 
needed. More flexibility will be provided in future versions.


Amortized Variational Inference
'''''''''''''''''''''''''''''''

We briefly describe some key components of SymJAX.

.. _basic_avi:

The principles of AVI
---------------------



Reinforcement Learning
''''''''''''''''''''''

We briefly describe some key components of SymJAX.

.. _rl_notations:

Notations
---------

- immediate reward :math:`r_t` is observed from the environment at state :math:`𝑠_{t}` by performing action :math:`𝑎_{t}`

- total discounted reward :math:`𝐺_t(γ)` often abbreviated as :math:`𝐺_t` and defined as

  .. math::
      𝐺_t = Σ_{t'=t+1}^{T}γ^{t'-t-1}r_t

- action-value function :math:`Q_{π}(𝑠,𝑎)` is the expected return starting from state 𝑠, following policy 𝜋 and taking action 𝑎

  .. math::
      Q_{π}(𝑠,𝑎)=E_{π}[𝐺_{t}|𝑠_{t} = 𝑠,𝑎_{t}=𝑎]

- state-value function :math:`V_{π}(𝑠)` is the expected return starting from state 𝑠 following policy 𝜋 as in

  .. math::
      V_{π}(𝑠)&=E_{π}[𝐺_{t}|𝑠_{t} = 𝑠]\\
            &=Σ_{𝑎 ∈ 𝐴}π(𝑎|𝑠)Q_{π}(𝑠,𝑎)

  in a deterministic policy setting, one has directly :math:`V_{π}(𝑠)=Q_{π}(𝑠,π(𝑠))`.
  in a greedy policy one might have :math:`V^{*}_{π}(𝑠)=\max_{𝑎∈𝐴}Q_{π}(𝑠,𝑎)` where :math:`V^{*}_{π}` is the best value of a state if you could follow an (unknown) optimum policy.

- TD-error

    + :math:`𝛿_t=r_t+γQ(𝑠_{t+1},𝑎_{t+1})-Q(𝑠_{t},𝑎_{t})`

- advantage value : how much better it is to take a specific action compared to the average at the given state

  .. math::
      A(s_t,𝑎_t)&=Q(𝑠_t,𝑎_t)-V(𝑠_t)\\
      A(𝑠_t,𝑎_t)&=E[r_{t+1}+ γ V(𝑠_{t+1})]-V(𝑠_t)\\
      A(𝑠_t,𝑎_t)&=r_{t+1}+ γ V(𝑠_{t+1})-V(𝑠_t)

  The formulation of policy gradients with advantage functions is extremely common, and there are `many different ways <https://arxiv.org/abs/1506.02438>`_ of estimating the advantage function used by different algorithms.

- probability of a trajectory :math:`τ=(s_0,a_0,...,s_{T+1})` is given by

  .. math::
      p(τ|θ)=p_{0}(s_0)Π_{t=0}^{T}p(𝑠_{t+1}|𝑠_{t},𝑎_{t})π_{0}(𝑎_{t}|𝑠_{t})


Policy gradient and REINFORCE
-----------------------------

`Policy gradient and REINFORCE <https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63>`_ : Policy gradient methods are ubiquitous in model free reinforcement learning algorithms — they appear frequently in reinforcement learning algorithms, especially so in recent publications. The policy gradient method is also the “actor” part of Actor-Critic methods. Its implementation (REINFORCE) is also known as Monte Carlo Policy Gradients. Policy gradient methods update the probability distribution of actions :math:`π(a|s)` so that actions with higher expected reward have a higher probability value for an observed state.

    + needs to reach end of episode to compute discounted rewards and train the model
    + only needs an actor (a.k.a policy) network
    + noisy gradients and high variance => instability and slow convergence
    + fails for trajectories having a cumulative reward of 0

Tricks
------

- `normalizing discounter rewards (or advantages) <http://arxiv.org/abs/1506.02438>`_ : In practice it can can also be important to normalize these. For example, suppose we compute [discounted cumulative reward] for all of the 20,000 actions in the batch of 100 Pong game rollouts above. One good idea is to “standardize” these returns (e.g. subtract mean, divide by standard deviation) before we plug them into backprop. This way we’re always encouraging and discouraging roughly half of the performed actions. Mathematically you can also interpret these tricks as a way of controlling the variance of the policy gradient estimator.
