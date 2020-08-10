# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
import gym

# https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44


class ddpg:
    def __init__(
        self, env, actor, critic, gamma=0.99, tau=0.01, lr=1e-3, batch_size=32, n=1,
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

        # state
        state = T.Placeholder((batch_size, num_states), "float32")
        next_state = T.Placeholder((batch_size, num_states), "float32")
        rewards = T.Placeholder((batch_size,), "float32")
        terminal = T.Placeholder((batch_size,), "float32")
        action = T.Placeholder((batch_size, num_actions), "float32")

        with symjax.Scope("actor_critic"):
            with symjax.Scope("actor"):
                suggested_actions = action_min + (action_max - action_min) * actor(
                    state
                )
            with symjax.Scope("critic"):
                q_values_of_given_actions = critic(state, action)

        q_values_of_suggested_actions = q_values_of_given_actions.clone(
            {action: suggested_actions}
        )

        with symjax.Scope("actor_critic_target"):
            slow_target_next_actions = action_min + (action_max - action_min) * actor(
                next_state
            )
            slow_q_values_next = critic(next_state, slow_target_next_actions)

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        discount = self.gamma ** self.n
        y_i = rewards + discount * (1 - terminal) * slow_q_values_next

        actor_loss = -q_values_of_suggested_actions.mean()
        critic_loss = T.mean((q_values_of_given_actions - y_i) ** 2)

        nn.optimizers.Adam(
            actor_loss, lr, params=symjax.get_variables(scope="/actor_critic/actor/*"),
        )
        nn.optimizers.Adam(
            critic_loss,
            lr,
            params=symjax.get_variables(scope="/actor_critic/critic/*"),
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

        self.critic_predict = symjax.function(
            state, action, outputs=q_values_of_given_actions
        )

        t_params = symjax.get_variables(scope="/actor_critic_target/*")
        params = symjax.get_variables(scope="/actor_critic/*")

        self.update_target = symjax.function(
            updates={t: tau * e + (1 - tau) * t for t, e in zip(t_params, params)}
        )

        single_state = T.Placeholder((1, num_states), "float32")
        if not continuous:
            scaled_out = clean_action.argmax(-1)

        self.act = symjax.function(
            single_state, outputs=suggested_actions.clone({state: single_state})[0],
        )

    def train(self, buffer, *args, **kwargs):
        import time

        ti = time.time()
        (s, a, r, s2, t,) = buffer.sample(self.batch_size)

        if not self.continuous:
            a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        if buffer.with_priorities:
            td_error = np.abs(y_i - self.critic_predict(s, a))
            buffer.update_priorities(buffer.current_indices, td_error)

        a_loss, q_loss = self.update(s, a, r, t, s2)

        self.update_target()
        return a_loss, q_loss
