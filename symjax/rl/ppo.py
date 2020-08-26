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
        entcoeff=0.01,
        target_kl=0.01,
        train_pi_iters=4,
        train_v_iters=4,
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
        self.entcoeff = entcoeff

        state_ph = T.Placeholder((batch_size, num_states), "float32")
        ret_ph = T.Placeholder((batch_size,), "float32")
        adv_ph = T.Placeholder((batch_size,), "float32")
        act_ph = T.Placeholder((batch_size, num_actions), "float32")

        with symjax.Scope("actor"):
            logits = actor(state_ph)
            if continuous:
                logstd = T.Variable(
                    -0.5 * np.ones(num_actions, dtype=np.float32),
                    name="logstd",
                )
        with symjax.Scope("old_actor"):
            old_logits = actor(state_ph)
            if continuous:
                old_logstd = T.Variable(
                    -0.5 * np.ones(num_actions, dtype=np.float32),
                    name="logstd",
                )

        if not continuous:
            pi = Categorical(logits=logits)
        else:
            pi = MultivariateNormal(mean=logits, diag_log_std=logstd)

        actions = T.clip(pi.sample(), -2, 2)  # pi

        actor_params = actor_params = symjax.get_variables(scope="/actor/")
        old_actor_params = actor_params = symjax.get_variables(scope="/old_actor/")

        self.update_target = symjax.function(
            updates={o: a for o, a in zip(old_actor_params, actor_params)}
        )

        # PPO objectives
        # pi(a|s) / pi_old(a|s)
        pi_log_prob = pi.log_prob(act_ph)
        old_pi_log_prob = pi_log_prob.clone({logits: old_logits, logstd: old_logstd})

        ratio = T.exp(pi_log_prob - old_pi_log_prob)
        surr1 = ratio * adv_ph
        surr2 = adv_ph * T.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

        pi_loss = -T.minimum(surr1, surr2).mean()
        # ent_loss = pi.entropy().mean() * self.entcoeff

        with symjax.Scope("critic"):
            v = critic(state_ph)
        # critic loss
        v_loss = ((ret_ph - v) ** 2).mean()

        # Info (useful to watch during learning)

        # a sample estimate for KL
        approx_kl = (old_pi_log_prob - pi_log_prob).mean()
        # a sample estimate for entropy
        # approx_ent = -logprob_given_actions.mean()
        # clipped = T.logical_or(
        #     ratio > (1 + clip_ratio), ratio < (1 - clip_ratio)
        # )
        # clipfrac = clipped.astype("float32").mean()

        with symjax.Scope("update_pi"):
            print(len(actor_params), "actor parameters")
            nn.optimizers.Adam(
                pi_loss,
                self.lr,
                params=actor_params,
            )
        with symjax.Scope("update_v"):
            critic_params = symjax.get_variables(scope="/critic/")
            print(len(critic_params), "critic parameters")

            nn.optimizers.Adam(
                v_loss,
                self.lr,
                params=critic_params,
            )

        self.get_params = symjax.function(outputs=critic_params)

        self.learn_pi = symjax.function(
            state_ph,
            act_ph,
            adv_ph,
            outputs=[pi_loss, approx_kl],
            updates=symjax.get_updates(scope="/update_pi/"),
        )
        self.learn_v = symjax.function(
            state_ph,
            ret_ph,
            outputs=v_loss,
            updates=symjax.get_updates(scope="/update_v/"),
        )

        single_state = T.Placeholder((1, num_states), "float32")
        single_v = v.clone({state_ph: single_state})
        single_sample = actions.clone({state_ph: single_state})

        self._act = symjax.function(single_state, outputs=single_sample)
        self._get_v = symjax.function(single_state, outputs=single_v)

        single_action = T.Placeholder((1, num_actions), "float32")
        # self.get_kl = symjax.function(
        #     single_state,
        #     single_action,
        #     outputs=logprob_given_actions.clone(
        #         {state_ph: single_state, act_ph: single_action}
        #     ),
        # )

    def choose_action(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self._act(s)[0]

    def get_value(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self._get_v(s)

    def train(self, buffer, *args, **kwargs):

        self.update_target()

        (s, a, r, adv) = buffer.sample(
            buffer.length,
            ["state", "action", "advantage", "reward-to-go"],
        )

        if not self.continuous:
            a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        r -= r.mean()
        r /= r.std()
        # r = np.maximum(np.minimum(r, 1), -1)

        # print(self.get_kl(s[[0]], a[[0]]))
        # print(self.get_KL(s, a))
        # print(self.get_kl(s[[0]], a[[0]]))
        # print(self.get_KL(s, a))
        # print(old_logprobs[0])
        # adsf

        for _ in range(self.train_pi_iters):
            losses = []
            for s1, a1, adv1 in symjax.data.utils.batchify(
                s,
                a,
                adv,
                batch_size=self.batch_size,
                option="random_see_all",
            ):
                loss = self.learn_pi(s1, a1, adv1)
                # losses.append(loss)
                # if kl > 1.5 * self.target_kl:
                # break
        for _ in range(self.train_v_iters):
            losses = []
            for s1, r1 in symjax.data.utils.batchify(
                s,
                r,
                batch_size=self.batch_size,
                option="random_see_all",
            ):

                losses.append(self.learn_v(s1, r1))
