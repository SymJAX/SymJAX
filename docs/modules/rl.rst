.. _symjax-rl:

:mod:`symjax.rl`
---------------------------



**Notions**

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


**Models**

    - `Policy gradient and REINFORCE <https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63>`_ : Policy gradient methods are ubiquitous in model free reinforcement learning algorithms — they appear frequently in reinforcement learning algorithms, especially so in recent publications. The policy gradient method is also the “actor” part of Actor-Critic methods. Its implementation (REINFORCE) is also known as Monte Carlo Policy Gradients. Policy gradient methods update the probability distribution of actions :math:`π(a|s)` so that actions with higher expected reward have a higher probability value for an observed state.

        + needs to reach end of episode to compute discounted rewards and train the model
        + only needs an actor (a.k.a policy) network
        + noisy gradients and high variance => instability and slow convergence
        + fails for trajectories having a cumulative reward of 0

**Tricks**

    - `normalizing discounter rewards (or advantages) <http://arxiv.org/abs/1506.02438>`_ : In practice it can can also be important to normalize these. For example, suppose we compute [discounted cumulative reward] for all of the 20,000 actions in the batch of 100 Pong game rollouts above. One good idea is to “standardize” these returns (e.g. subtract mean, divide by standard deviation) before we plug them into backprop. This way we’re always encouraging and discouraging roughly half of the performed actions. Mathematically you can also interpret these tricks as a way of controlling the variance of the policy gradient estimator.


Implementation of basic agents, environment utilites and learning policies

.. automodule:: symjax.rl.utils


..  autosummary::

	Buffer
	run
	

.. automodule:: symjax.rl.agents


..  autosummary::

	Actor
	Critic


.. automodule:: symjax.rl

..  autosummary::
	REINFORCE
	ActorCritic
	PPO
	DDPG





Detailed Descriptions
=====================

.. automodule:: symjax.rl.utils

.. autoclass:: Buffer
.. autofunction:: run

.. automodule:: symjax.rl.agents

.. autoclass:: Actor
.. autoclass:: Critic

.. automodule:: symjax.rl

.. autoclass:: REINFORCE
.. autoclass:: ActorCritic
.. autoclass:: PPO
.. autoclass:: DDPG