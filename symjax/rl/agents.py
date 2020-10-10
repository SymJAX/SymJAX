import symjax
import symjax.tensor as T
import numpy as np


class Agent(object):
    def get_action(self, state):
        return self.actor.get_action(state)

    def get_actions(self, states):
        return self.actor.get_actions(states)

    def play(self, state, env, skip_frames=1, reward_scaling=1):

        action, extra = self.get_action(state)

        if hasattr(self, "critic"):
            if hasattr(self.critic, "actions"):
                value = self.critic.get_q_value(state, action)
            else:
                value = self.critic.get_q_value(state)
            extra.update({"V": value})

        reward = 0
        for k in range(skip_frames):
            next_state, r, terminal, info = env.step(action)
            reward += r
            if terminal:
                break
        reward /= skip_frames
        reward *= reward_scaling

        base = {
            "state": state,
            "action": action,
            "reward": reward,
            "next-state": next_state,
            "terminal": terminal,
        }

        return next_state, terminal, {**base, **extra}

    def update_target(self, tau=None):

        if not hasattr(self, "_update_target"):
            with symjax.Scope("update_target"):
                targets = []
                currents = []
                if hasattr(self, "target_actor"):
                    targets += self.target_actor.params(True)
                    currents += self.actor.params(True)
                if hasattr(self, "target_critic"):
                    targets += self.target_critic.params(True)
                    currents += self.critic.params(True)

                _tau = T.Placeholder((), "float32")
                updates = {
                    t: t * (1 - _tau) + a * _tau for t, a in zip(targets, currents)
                }
            self._update_target = symjax.function(_tau, updates=updates)

        if tau is None:
            if not hasattr(self, "tau"):
                raise RuntimeError("tau must be specified")
        tau = tau or self.tau
        self._update_target(tau)


class Actor(object):
    """actor (state to action mapping) for RL

    This class implements an actor. The user must first define its own class
    inheriting from  :py:class:`Actor` and implementing only the
    `create_network` method. This method will then be used internally to
    instantiace the actor network.

    If the used distribution is `symjax.probabilities.Normal` then the output
    of the `create_network` method should be first the mean and then the
    `covariance`.

    Notes:
    ------

    In general the user should not instanciate this class, instead pass the
    user's inherited class (uninstanciated) to a policy-learning method.

    Parameters:
    -----------

    states: Tensor-like
        the states of the environment (batch size in first axis)

    batch_size: int
        the batch size

    actions_distribution: None or symjax.probabilities.Distribution object
        the distribution for the actions, if the policy is deterministic, then
        put this to `None`. Note, this is different than the noise parameter
        employed for exploration, this is simply the rv modeling of the
        actions used to compute probabilities of sampled actions and
        the likes

    """

    def __init__(self, states, actions_distribution=None, name="actor"):

        self.state_shape = states.shape[1:]
        state = T.Placeholder((1,) + states.shape[1:], "float32")
        self.actions_distribution = actions_distribution

        with symjax.Scope(name):
            if actions_distribution == symjax.probabilities.Normal:

                means, covs = self.create_network(states)

                actions = actions_distribution(means, cov=covs)
                samples = actions.sample()
                samples_log_prob = actions.log_prob(samples)

                action = symjax.probabilities.MultivariateNormal(
                    means.clone({states: state}),
                    cov=covs.clone({states: state}),
                )
                sample = self.action.sample()
                sample_log_prob = self.action.log_prob(sample)

                self._get_actions = symjax.function(
                    states, outputs=[samples, samples_log_prob]
                )
                self._get_action = symjax.function(
                    state,
                    outputs=[sample[0], sample_log_prob[0]],
                )
            elif actions_distribution is None:
                actions = self.create_network(states)
                action = actions.clone({states: state})

                self._get_actions = symjax.function(states, outputs=actions)
                self._get_action = symjax.function(state, outputs=action[0])

            self._params = symjax.get_variables(
                trainable=None, scope=symjax.current_graph().scope_name
            )
        self.actions = actions
        self.state = state
        self.action = action

    def params(self, trainable):
        if trainable is None:
            return self._params
        return [p for p in self._params if p.trainable == trainable]

    def get_action(self, state):
        if state.ndim == len(self.state_shape):
            state = state[np.newaxis, :]
        if not hasattr(self, "_get_action"):
            raise RuntimeError("actor not well initialized")
        if self.actions_distribution is None:
            return self._get_action(state), {}
        else:
            a, probs = self._get_action(state)
            return a, {"log_probs": probs}

    def get_actions(self, state):
        if not hasattr(self, "_get_actions"):
            raise RuntimeError("actor not well initialized")
        if self.actions_distribution is None:
            return self._get_actions(state), {}
        else:
            a, probs = self._get_actions(state)
            return a, {"log_probs": probs}

    def create_network(self, states, action_shape):
        """creating of the actor network returning the actions

        This method has to be implemented by the user in a own actor class
        inheriting from `symjax.rl.Actor`. This method should take
        two arguments, the states and the action_dim, and return
        the actions after a possible nonlinear transformation of the given
        states by say a deep networks

        Parameters:
        -----------

        states: Tensor
            the states with shape (batch_size, *state_shape)

        action_shape: tuple or list
            the shape of a (single) action, for example in classical
            pendulum this would be `(2,)`.

        Returns:
        --------

        actions: Tensor
            the actions with shape (batch_size, *action_shape)

        """
        raise RuntimeError("Not implemented, user should define its own")


class Critic(object):
    def __init__(self, states, actions=None):
        self.state_shape = states.shape[1:]
        state = T.Placeholder((1,) + states.shape[1:], "float32", name="critic_state")
        if actions:
            self.action_shape = actions.shape[1:]
            action = T.Placeholder(
                (1,) + actions.shape[1:], "float32", name="critic_action"
            )
            action_shape = action.shape[1:]

            with symjax.Scope("critic"):
                q_values = self.create_network(states, actions)
                if q_values.ndim == 2:
                    assert q_values.shape[1] == 1
                    q_values = q_values[:, 0]
                q_value = q_values.clone({states: state, actions: action})
                self._params = symjax.get_variables(
                    trainable=None, scope=symjax.current_graph().scope_name
                )

            inputs = [states, actions]
            input = [state, action]
            self.actions = actions
            self.action = action

        else:
            with symjax.Scope("critic"):
                q_values = self.create_network(states)
                if q_values.ndim == 2:
                    assert q_values.shape[1] == 1
                    q_values = q_values[:, 0]
                q_value = q_values.clone({states: state})
                self._params = symjax.get_variables(
                    trainable=None, scope=symjax.current_graph().scope_name
                )

            inputs = [states]
            input = [state]

        self.q_values = q_values
        self.state = state
        self.states = states

        self._get_q_values = symjax.function(*inputs, outputs=q_values)
        self._get_q_value = symjax.function(*input, outputs=q_value[0])

    def params(self, trainable):
        if trainable is None:
            return self._params
        return [p for p in self._params if p.trainable == trainable]

    def get_q_value(self, state, action=None):
        if state.ndim == len(self.state_shape):
            state = state[np.newaxis, :]
        if action is not None:
            if action.ndim == len(self.action_shape):
                action = action[np.newaxis, :]
        if not hasattr(self, "_get_q_value"):
            raise RuntimeError("critic not well initialized")
        if action is None:
            return self._get_q_value(state)
        else:
            return self._get_q_value(state, action)

    def get_q_values(self, states, actions=None):
        if not hasattr(self, "_get_q_values"):
            raise RuntimeError("critic not well initialized")
        if actions is not None:
            return self._get_q_values(states, actions)
        else:
            return self._get_q_values(states)

    def create_network(self, states, actions=None):
        raise RuntimeError("Not implemented, user should define its own")


class OrnsteinUhlenbeckProcess:
    """dXt = theta*(mu-Xt)*dt + sigma*dWt"""

    def __init__(
        self,
        mean=0.0,
        std_dev=0.2,
        theta=0.15,
        dt=1e-2,
        noise_decay=0.99,
        initial_noise_scale=1,
        init=None,
    ):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = (dt,)
        self.init = init
        self.noise_decay = noise_decay
        self.initial_noise_scale = initial_noise_scale
        self.end_episode()

    def __call__(self, action, episode):

        with symjax.Scope("OUProcess"):
            self.episode = T.Variable(1, "float32", name="episode", trainable=False)

        self.noise_scale = self.initial_noise_scale * self.noise_decay ** episode

        x = (
            self.process
            + self.theta * (self.mean - self.process) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=action.shape)
        )
        # Store x into process
        # Makes next noise dependent on current one
        self.process = x

        return action + self.noise_scale * self.process

    def end_episode(self):
        if self.init is None:
            self.process = np.zeros(1)
        else:
            self.process = self.init
