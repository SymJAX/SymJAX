# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
import gym

# https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44


class ddpg:
    def __init__(
        self, env, actor, critic, gamma=0.99, tau=0.01, lr=1e-4, batch_size=32, n=1,
    ):

        # get size of state space and action space
        num_states = env.observation_space.shape[0]
        continuous = type(env.action_space) == gym.spaces.box.Box

        if continuous:
            num_actions = env.action_space.shape[0]
            action_min = env.action_space.low
            action_max = env.action_space.high
        else:
            num_actions = env.action_space.n
            action_max = 1

        self.batch_size = batch_size
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_dim = (batch_size, num_states)
        self.action_dim = (batch_size, num_actions)
        self.gamma = gamma
        self.continuous = continuous
        self.n = n
        self.extras = {}
        self.tau = tau

        # state
        state = T.Placeholder((batch_size, num_states), "float32")
        next_state = T.Placeholder((batch_size, num_states), "float32")
        rewards = T.Placeholder((batch_size,), "float32")
        terminal = T.Placeholder((batch_size,), "float32")
        action = T.Placeholder((batch_size, num_actions), "float32")

        with symjax.Scope("critic"):
            critic_value = critic(state, action)

        with symjax.Scope("target_actor"):
            target_action = actor(next_state)

        with symjax.Scope("target_critic"):

            # One step TD targets y_i for (s,a) from experience replay
            # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
            # = r_i if s' terminal
            # also refered as Bellman backup for Q function
            y_i = rewards + self.gamma * (1 - terminal) * critic(
                next_state, target_action
            )

        # DDPG losses
        critic_loss = nn.losses.squared_differences(critic_value, y_i).mean()

        critic_params = symjax.get_variables(scope="/critic/")
        print(len(critic_params), "critic parameters")
        nn.optimizers.Adam(
            critic_loss, lr, params=critic_params,
        )

        with symjax.Scope("actor"):
            actions = actor(state)

        critic_value = critic_value.clone({action: actions})
        actor_loss = -critic_value.mean()
        actor_params = symjax.get_variables(scope="/actor/")
        print(len(actor_params), "actor parameters")
        nn.optimizers.Adam(
            actor_loss, lr, params=actor_params,
        )

        self.update = symjax.function(
            state,
            action,
            rewards,
            terminal,
            next_state,
            outputs=[actor_loss, critic_loss],
            updates=symjax.get_updates(),
        )

        t_params = symjax.get_variables(scope="/target_actor/") + symjax.get_variables(
            scope="/target_critic/"
        )
        params = symjax.get_variables(scope="/actor/") + symjax.get_variables(
            scope="/critic/"
        )

        _tau = T.Placeholder((), "float32")
        self.update_target = symjax.function(
            _tau,
            updates={t: _tau * e + (1 - _tau) * t for t, e in zip(t_params, params)},
        )
        self.update_target(1)

        single_state = T.Placeholder((1, num_states), "float32")
        if not continuous:
            scaled_out = clean_action.argmax(-1)

        self._act = symjax.function(
            single_state, outputs=actions.clone({state: single_state})[0],
        )

    def act(self, state):
        action = self._act(state)
        return action, {"V": 0}

    def train(self, buffer, *args, **kwargs):

        s, a, r, s2, t = buffer.sample(self.batch_size)

        if not self.continuous:
            a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        a_loss, q_loss = self.update(s, a, r, t, s2)

        self.update_target(self.tau)
        return a_loss, q_loss
