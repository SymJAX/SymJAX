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

# ===========================
#   Set rewards
# ===========================


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_states,
        Q,
        learning_rate=0.01,
        reward_decay=0.8,
        e_greedy=0.9,
        replace_target_iter=30,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=0.001,
        save_steps=-1,
        output_graph=False,
        record_history=True,
        observation_interval=0.01,
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.record_history = record_history
        self.observation_interval = observation_interval

        # total learning step
        self.learn_step_counter = 0
        self.action_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_states * 2 + 2))
        self.memory_counter = 0
        # consist of [target_net, evaluate_net]
        self.build_net(Q)

        # save data
        self.save_steps = save_steps
        self.steps = 0

        t_params = symjax.get_variables(scope="target_net")
        e_params = symjax.get_variables(scope="eval_net")

        replacement = {t: e for t, e in zip(t_params, e_params)}
        self.replace = symjax.function(updates=replacement)

        self.cost_history = []
        self._tmp_cost_history = []

    def build_net(self, Q):
        # ------------------ all inputs ------------------------
        state = T.Placeholder([self.batch_size, self.n_states], "float32", name="s")
        next_state = T.Placeholder(
            [self.batch_size, self.n_states], "float32", name="s_"
        )
        reward = T.Placeholder(
            [
                self.batch_size,
            ],
            "float32",
            name="r",
        )  # input reward
        action = T.Placeholder(
            [
                self.batch_size,
            ],
            "int32",
            name="a",
        )  # input Action

        with symjax.Scope("eval_net"):
            q_eval = Q(state, self.n_actions)
        with symjax.Scope("test_set"):
            q_next = Q(next_state, self.n_actions)

        q_target = reward + self.reward_decay * q_next.max(1)
        q_target = T.stop_gradient(q_target)

        a_indices = T.stack([T.range(self.batch_size), action], axis=1)
        q_eval_wrt_a = T.take_along_axis(q_eval, action.reshape((-1, 1)), 1).squeeze(1)
        loss = T.mean((q_target - q_eval_wrt_a) ** 2)
        nn.optimizers.Adam(loss, self.lr)

        self.train = symjax.function(
            state, action, reward, next_state, updates=symjax.get_updates()
        )
        self.q_eval = symjax.function(state, outputs=q_eval)

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def act(self, state, memory):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the state and get q value for every actions
            actions_logprobs = self.q_eval(
                state.repeat(self.batch_size, 0).astype("float32")
            )[0]
            action = np.argmax(actions_logprobs)
        else:
            action = np.random.randint(0, self.n_actions)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprobs)

        self.action_step_counter += 1
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace()
            print("\ntarget_params_replaced\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        cost = self.train(
            batch_memory[:, : self.n_states],
            batch_memory[:, self.n_states].astype("int32"),
            batch_memory[:, self.n_states + 1],
            batch_memory[:, -self.n_states :],
        )

        self.steps += 1

        # increasing epsilon
        self.epsilon = (
            self.epsilon + self.epsilon_increment
            if self.epsilon < self.epsilon_max
            else self.epsilon_max
        )
        self.learn_step_counter += 1
