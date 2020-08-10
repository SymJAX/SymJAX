# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
import gym

# https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44


class ddpg:
    def __init__(
        self,
        env,
        actor,
        critic,
        gamma=0.99,
        tau=0.01,
        lr=1e-3,
        batch_size=32,
        epsilon=0.1,
        epsilon_decay=1 / 1000,
        min_epsilon=0.01,
        n=1,
    ):

        # get size of state space and action space
        num_states = env.observation_space.shape[0]
        continuous = type(env.action_space) == gym.spaces.box.Box

        if continuous:
            num_actions = env.action_space.shape[0]
            action_max = env.action_space.high[0]
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
        gradients = T.Placeholder((batch_size, num_actions), "float32")
        action = T.Placeholder((batch_size, num_actions), "float32")

        with symjax.Scope("actor_critic"):

            scaled_out = action_max * actor(state)
            Q = critic(state, action)

        with symjax.Scope("actor_critic_target"):
            scaled_out_target = action_max * actor(next_state)
            target_q = critic(next_state, scaled_out_target)

        Q_target = target_q.clone({next_state: state, scaled_out_target: action})

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        discount = self.gamma ** self.n
        y_i = rewards + discount * (1 - terminal) * target_q

        a_loss = -T.sum(gradients * scaled_out)
        q_loss = T.mean((Q - y_i) ** 2)

        nn.optimizers.Adam(a_loss + q_loss, lr)

        self.update = symjax.function(
            state,
            action,
            rewards,
            terminal,
            next_state,
            gradients,
            outputs=[a_loss, q_loss],
            updates=symjax.get_updates(),
        )
        g = symjax.gradients(T.mean(Q), [action])[0]
        self.get_gradients = symjax.function(state, action, outputs=g)

        self.actor_predict = symjax.function(state, outputs=scaled_out)
        self.actor_predict_target = symjax.function(
            next_state, outputs=scaled_out_target
        )
        self.critic_predict = symjax.function(state, action, outputs=Q)
        self.critic_predict_target = symjax.function(state, action, outputs=Q_target)

        t_params = symjax.get_variables(scope="/actor_critic_target/*")
        params = symjax.get_variables(scope="/actor_critic/*")
        replacement = {t: tau * e + (1 - tau) * t for t, e in zip(t_params, params)}
        self.update_target = symjax.function(updates=replacement)

        single_state = T.Placeholder((1, num_states), "float32")
        if not continuous:
            scaled_out = clean_action.argmax(-1)

        self.act = symjax.function(
            single_state, outputs=scaled_out.clone({state: single_state})[0]
        )

    def train(self, buffer, *args, **kwargs):
        (s_batch, a_batch, r_batch, s2_batch, t_batch,) = buffer.sample(self.batch_size)

        if not self.continuous:
            a_batch = (np.arange(self.num_actions) == a_batch[:, None]).astype(
                "float32"
            )

        if buffer.with_priorities:
            td_error = np.abs(y_i - self.critic_predict(s_batch, a_batch))
            buffer.update_priorities(buffer.current_indices, td_error)

        # Update the critic given the targets
        gradients = self.get_gradients(s_batch, a_batch)
        a_loss, q_loss = self.update(
            s_batch, a_batch, r_batch, t_batch, s2_batch, gradients
        )

        # Update target networks
        self.update_target()
        return a_loss, q_loss
