# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
from symjax.probabilities import Categorical, MultivariateNormal
import math
import matplotlib.pyplot as plt
from collections import deque
import gym
import random


import scipy.signal

# https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        gamma,
        K_epochs,
        eps_clip,
        actor,
        critic,
        batch_size,
        continuous=True,
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        state = T.Placeholder((batch_size,) + state_dim, "float32")

        reward = T.Placeholder((batch_size,), "float32")
        old_action_logprobs = T.Placeholder((batch_size,), "float32")

        logits = actor(state)

        if not continuous:
            given_action = T.Placeholder((batch_size,), "int32")
            dist = Categorical(logits=logits)
        else:
            mean = T.tanh(logits[:, : logits.shape[1] // 2])
            std = T.exp(logits[:, logits.shape[1] // 2 :])
            given_action = T.Placeholder((batch_size, action_dim), "float32")
            dist = MultivariateNormal(mean=mean, diag_std=std)

        sample = dist.sample()
        sample_logprobs = dist.log_prob(sample)

        self._act = symjax.function(state, outputs=[sample, sample_logprobs])

        given_action_logprobs = dist.log_prob(given_action)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = T.exp(sample_logprobs - old_action_logprobs)
        ratios = T.clip(ratios, None, 1 + self.eps_clip)

        state_value = critic(state)
        advantages = reward - T.stop_gradient(state_value)

        loss = (
            -T.mean(ratios * advantages)
            + 0.5 * T.mean((state_value - reward) ** 2)
            - 0.0 * dist.entropy().mean()
        )

        print(loss)
        nn.optimizers.Adam(loss, self.lr)

        self.learn = symjax.function(
            state,
            given_action,
            reward,
            old_action_logprobs,
            outputs=T.mean(loss),
            updates=symjax.get_updates(),
        )

    def act(self, state, memory):
        action, action_logprobs = self._act(state[None, :].repeat(self.batch_size, 0))
        memory.states.append(state)
        memory.actions.append(action[0])
        memory.logprobs.append(action_logprobs[0])
        return action

    def train(self):

        (s_batch, a_batch, r_batch, s2_batch, t_batch, batch,) = self.env.buffer.sample(
            self.batch_size
        )
        # Calculate targets
        target_q = self.critic_predict_target(
            s2_batch, self.actor_predict_target(s2_batch)
        )

        y_i = r_batch + self.gamma * (1 - t_batch[:, None]) * target_q

        if not self.continuous:
            a_batch = (np.arange(self.num_actions) == a_batch[:, None]).astype(
                "float32"
            )

        td_error = np.abs(y_i - self.critic_predict(s_batch, a_batch))
        self.env.buffer.update_priorities(batch, td_error)

        # Update the critic given the targets
        q_loss, predicted_q_value = self.train_critic(s_batch, a_batch, y_i)

        # Update the actor policy using the sampled gradient
        a_outs = self.actor_predict(s_batch)
        # grads = self.get_action_grads(s_batch, a_outs)
        a_loss = self.train_actor(s_batch)  # , grads)

        # Update target networks
        self.update_target()
        return a_loss, q_loss

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = np.array(rewards)

        # Normalizing the rewards:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = memory.states
        old_actions = memory.actions
        old_logprobs = memory.logprobs

        # Optimize policy for K epochs:
        for _ in range(4):
            loss = self.learn(old_states, old_actions, rewards, old_logprobs)
            print("loss", loss)

        # Copy new weights into old policy:
