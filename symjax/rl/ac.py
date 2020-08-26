# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
from symjax.probabilities import Categorical, MultivariateNormal
import gym


class ac:
    """actor critic,"""

    def __init__(
        self,
        env,
        actor,
        critic,
        lr=1e-4,
        batch_size=32,
        train_pi_iters=80,
        train_v_iters=80,
    ):

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
        self.continuous = continuous
        self.lr = lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.extras = {}

        state_ph = T.Placeholder((batch_size, num_states), "float32")
        rew_ph = T.Placeholder((batch_size,), "float32")

        with symjax.Scope("actor"):
            logits = actor(state_ph)
            if not continuous:
                pi = Categorical(logits=logits)
            else:
                logstd = T.Variable(
                    -0.5 * np.ones(num_actions, dtype=np.float32),
                    name="logstd",
                )
                pi = MultivariateNormal(mean=logits, diag_log_std=logstd)

            actions = pi.sample()  # pi
            actions_log_prob = pi.log_prob(actions)  # logp

        with symjax.Scope("critic"):
            critic_value = critic(state_ph)

        # AC objectives
        diff = rew_ph - critic_value
        actor_loss = -(actions_log_prob * diff).mean()
        critic_loss = nn.losses.squared_differences(rew_ph, critic_value).mean()

        with symjax.Scope("update_pi"):
            nn.optimizers.Adam(
                actor_loss,
                self.lr,
                params=symjax.get_variables(scope="/actor/"),
            )
        with symjax.Scope("update_v"):
            nn.optimizers.Adam(
                critic_loss,
                self.lr,
                params=symjax.get_variables(scope="/critic/"),
            )

        self.learn_pi = symjax.function(
            state_ph,
            rew_ph,
            outputs=actor_loss,
            updates=symjax.get_updates(scope="/update_pi/"),
        )
        self.learn_v = symjax.function(
            state_ph,
            rew_ph,
            outputs=critic_loss,
            updates=symjax.get_updates(scope="/update_v/*"),
        )

        single_state = T.Placeholder((1, num_states), "float32")
        single_action = actions.clone({state_ph: single_state})[0]
        single_v = critic_value.clone({state_ph: single_state})

        self._act = symjax.function(
            single_state,
            outputs=[single_action, single_v],
        )

    def act(self, state):
        action, v = self._act(state)
        return action, {"V": v}

    def train(self, buffer, *args, **kwargs):

        s, r = buffer.sample(self.batch_size, ["state", "reward-to-go"])

        r -= r.mean()
        r /= r.std()

        if not self.continuous:
            a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        for _ in range(self.train_pi_iters):
            loss = self.learn_pi(s, r)

        for _ in range(self.train_v_iters):
            loss = self.learn_v(s, r)
