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
    def __init__(self, states, noise=None, distribution="deterministic"):

        self.states = states
        self.state = T.Placeholder((1,) + states.shape.get()[1:], "float32")
        self.state_shape = self.state.shape.get()[1:]
        self.distribution = distribution

        with symjax.Scope("actor") as q:
            if distribution == "gaussian":
                means, covs = self.create_network(self.states)
                self.actions = symjax.probabilities.MultivariateNormal(
                    means, diag_log_std=covs
                )
                self.samples = self.actions.sample()
                self.samples_log_prob = self.actions.log_prob(self.samples)
                mean = means.clone({self.states: self.state})
                cov = covs.clone({self.states: self.state})
                self.action = symjax.probabilities.MultivariateNormal(
                    mean, diag_log_std=cov
                )
                self.sample = self.action.sample()
                self.sample_log_prob = self.action.log_prob(self.sample)
                self._get_actions = symjax.function(
                    self.states, outputs=[self.samples, self.samples_log_prob]
                )
                self._get_action = symjax.function(
                    self.state,
                    outputs=[self.sample[0], self.sample_log_prob[0]],
                )
            else:
                self.actions = self.create_network(self.states)
                self.action = self.actions.clone({self.states: self.state})

                self._get_actions = symjax.function(self.states, outputs=self.actions)
                self._get_action = symjax.function(self.state, outputs=self.action[0])

        self._params = q.variables(trainable=None)

    def params(self, trainable):
        if trainable is None:
            return self._params
        return [p for p in self._params if p.trainable == trainable]

    def get_action(self, state):
        if state.ndim == len(self.state_shape):
            state = state[np.newaxis, :]
        if not hasattr(self, "_get_action"):
            raise RuntimeError("actor not well initialized")
        if self.distribution == "deterministic":
            return self._get_action(state), {}
        else:
            a, probs = self._get_action(state)
            return a, {"log_probs": probs}

    def get_actions(self, state):
        if not hasattr(self, "_get_actions"):
            raise RuntimeError("actor not well initialized")
        if self.distribution == "deterministic":
            return self._get_actions(state), {}
        else:
            a, probs = self._get_actions(state)
            return a, {"log_probs": probs}

    def create_network(self, states, action_dim):
        raise RuntimeError("Not implemented, user should define its own")


class Critic(object):
    def __init__(self, states, actions=None):
        self.states = states
        self.state = T.Placeholder(
            (1,) + states.shape.get()[1:], "float32", name="critic_state"
        )
        self.state_shape = self.state.shape.get()[1:]
        if actions:
            self.actions = actions
            self.action = T.Placeholder(
                (1,) + actions.shape.get()[1:], "float32", name="critic_action"
            )
            self.action_shape = self.action.shape.get()[1:]

            with symjax.Scope("critic") as q:
                self.q_values = self.create_network(self.states, self.actions)
                if self.q_values.ndim == 2:
                    assert self.q_values.shape.get()[1] == 1
                    self.q_values = self.q_values[:, 0]
                self.q_value = self.q_values.clone(
                    {self.states: self.state, self.actions: self.action}
                )
                self._params = q.variables(trainable=None)

            inputs = [self.states, self.actions]
            input = [self.state, self.action]

        else:
            with symjax.Scope("critic") as q:
                self.q_values = self.create_network(self.states)
                if self.q_values.ndim == 2:
                    assert self.q_values.shape.get()[1] == 1
                    self.q_values = self.q_values[:, 0]
                self.q_value = self.q_values.clone({self.states: self.state})
                self._params = q.variables(trainable=None)

            inputs = [self.states]
            input = [self.state]

        self._get_q_values = symjax.function(*inputs, outputs=self.q_values)
        self._get_q_value = symjax.function(*input, outputs=self.q_value[0])

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
