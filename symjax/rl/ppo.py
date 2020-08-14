# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
from symjax.probabilities import Categorical, MultivariateNormal
import gym


class ppo:
    """Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL
    """

    def __init__(
        self,
        env,
        actor,
        critic,
        lr=1e-4,
        batch_size=32,
        n=1,
        clip_ratio=0.2,
        target_kl=0.01,
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
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.extras = {"logprob": ()}

        state_ph = T.Placeholder((batch_size, num_states), "float32")
        ret_ph = T.Placeholder((batch_size,), "float32")
        adv_ph = T.Placeholder((batch_size,), "float32")
        act_ph = T.Placeholder((batch_size, num_actions), "float32")
        logp_old_ph = T.Placeholder((batch_size,), "float32")

        with symjax.Scope("actor"):
            logits = actor(state_ph)
            if not continuous:
                pi = Categorical(logits=logits)
            else:
                logstd = T.Variable(
                    -0.5 * np.ones(num_actions, dtype=np.float32), name="logstd",
                )
                pi = MultivariateNormal(mean=logits, diag_log_std=logstd)

            actions = pi.sample()  # pi
            logprob_actions = pi.log_prob(actions)  # logp_pi
            logprob_given_actions = pi.log_prob(act_ph)  # logp

        with symjax.Scope("critic"):
            v = critic(state_ph)

        # PPO objectives
        # pi(a|s) / pi_old(a|s)
        ratio = T.exp(logprob_given_actions - logp_old_ph)
        min_adv = T.where(adv_ph > 0, (1 + clip_ratio), (1 - clip_ratio)) * adv_ph
        pi_loss = -T.minimum(ratio * adv_ph, min_adv).mean()
        v_loss = ((ret_ph - v) ** 2).mean()

        # Info (useful to watch during learning)

        # a sample estimate for KL
        approx_kl = (logp_old_ph - logprob_given_actions).mean()
        # a sample estimate for entropy
        approx_ent = -logprob_given_actions.mean()
        clipped = T.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        clipfrac = clipped.astype("float32").mean()

        with symjax.Scope("update_pi"):
            actor_params = symjax.get_variables(scope="/actor/")
            print(len(actor_params), "actor parameters")
            nn.optimizers.Adam(
                pi_loss, self.lr, params=actor_params,
            )
        with symjax.Scope("update_v"):
            critic_params = symjax.get_variables(scope="/critic/")
            print(len(critic_params), "critic parameters")
            nn.optimizers.Adam(
                v_loss, self.lr, params=critic_params,
            )

        self.learn_pi = symjax.function(
            state_ph,
            act_ph,
            adv_ph,
            logp_old_ph,
            outputs=[pi_loss, approx_kl],
            updates=symjax.get_updates(scope="/update_pi/*"),
        )
        self.learn_v = symjax.function(
            state_ph,
            ret_ph,
            outputs=v_loss,
            updates=symjax.get_updates(scope="/update_v/*"),
        )

        single_state = T.Placeholder((1, num_states), "float32")

        single_action = actions.clone({state_ph: single_state})[0]
        single_logp_action = logprob_actions.clone({state_ph: single_state})[0]
        single_v = v.clone({state_ph: single_state})

        self._act = symjax.function(
            single_state, outputs=[single_action, single_v, single_logp_action],
        )
        single_action = T.Placeholder((1, num_actions), "float32")
        self.get_kl = symjax.function(
            single_state,
            single_action,
            outputs=logprob_given_actions.clone(
                {state_ph: single_state, act_ph: single_action}
            ),
        )
        self.get_KL = symjax.function(
            state_ph, act_ph, outputs=logprob_given_actions[0]
        )

    def act(self, state):
        action, v, logprob = self._act(state)
        return action, {"V": v, "logprob": logprob}

    def train(self, buffer, *args, **kwargs):

        (s, a, r, adv, old_logprobs) = buffer.sample(
            self.batch_size,
            ["state", "action", "advantage", "reward-to-go", "logprob"],
        )

        if not self.continuous:
            a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        r -= r.mean()
        r /= r.std()

        # print(self.get_kl(s[[0]], a[[0]]))
        # print(self.get_KL(s, a))
        # print(self.get_kl(s[[0]], a[[0]]))
        # print(self.get_KL(s, a))
        # print(old_logprobs[0])
        # adsf

        for _ in range(self.train_pi_iters):
            loss, kl = self.learn_pi(s, a, adv, old_logprobs)
            if kl > 1.5 * self.target_kl:
                print("Early stopping of pi learning at", _, kl, self.target_kl)
                break

        for _ in range(self.train_v_iters):
            loss_v = self.learn_v(s, r)
