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


class Reward(object):
    """reward management (discounts, n-step)

    Args:
    -----

    factor: float
        the amount of rescaling of the reward after discount has been
        applied.

    n: int or inf
        the n-step reward discount calculation, for the infinite case
        prefer giving ``np.inf``

    """

    def __init__(self, factor, gamma=None, n=8):
        # Reward parameters
        self.factor = factor
        self.n = n
        self.gamma = gamma

    # Set step rewards to total episode reward
    def total(self, ep_batch, tot_reward):
        for step in ep_batch:
            step[2] = tot_reward * self.factor
        return ep_batch

    # Set step rewards to discounted reward
    def discount(self, ep_batch):
        if self.n == 1:
            return ep_batch
        x = ep_batch[:, 2]

        if self.n == np.inf:
            b = [1]
            a = [1, -self.gamma]
        else:
            b = self.gamma ** np.arange(self.n)
            a = [1]
        discounted = scipy.signal.lfilter(b=b, a=a, x=x[::-1], axis=0)[::-1]
        discounted *= self.factor

        ep_batch[:, 2] = discounted

        return ep_batch


class BasicBuffer:
    def __init__(self, size):

        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    def clear(self):
        del self.buffer
        del self.priorities

        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    @property
    def len(self):
        return len(self.buffer)

    def push(self, *args):

        if not hasattr(self, "n_components"):
            self.n_components = len(args)
        else:
            assert len(args) == self.n_components

        self.buffer.append(args)
        self.priorities.append(1)

    def update_priorities(self, indices, priorities):
        for indx, priority in zip(indices, priorities):
            self.priorities[indx] = 1 + priority

    def sample(self, length, asarray=True, return_indices=False):

        assert len(self.buffer) > 0

        items = [[] for i in range(self.n_components)]

        indices = np.arange(len(self.buffer), dtype="int32")
        batch = random.choices(indices, weights=self.priorities, k=length)
        self.current_indices = batch

        for experience in batch:
            values = self.buffer[experience]
            for item, value in zip(items, values):
                item.append(value)

        if asarray:
            for i in range(len(items)):
                items[i] = np.asarray(items[i], dtype=np.float32)

        if return_indices:
            return items + [batch]
        else:
            return items


### Create both the actor and critic networks at once ###
### Q(s, mu(s)) returns the maximum Q for a given state s ###
class ddpg:
    def __init__(
        self,
        env_fn,
        actor,
        critic,
        gamma=0.99,
        tau=0.01,
        lr=1e-3,
        batch_size=32,
        epsilon=0.1,
        epsilon_decay=1 / 1000,
        min_epsilon=0.01,
        reward=None,
    ):

        # comment out this line if you don't want to record a video of the agent
        # if save_folder is not None:
        # test_env = gym.wrappers.Monitor(test_env)

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
        self.observ_min = np.clip(env.observation_space.low, -20, 20)
        self.observ_max = np.clip(env.observation_space.high, -20, 20)
        self.env = env
        self.reward = reward

        # state
        state = T.Placeholder((batch_size, num_states), "float32")
        gradients = T.Placeholder((batch_size, num_actions), "float32")
        action = T.Placeholder((batch_size, num_actions), "float32")
        target = T.Placeholder((batch_size, 1), "float32")

        with symjax.Scope("actor_critic"):

            scaled_out = action_max * actor(state)
            Q = critic(state, action)

        a_loss = -T.sum(gradients * scaled_out)
        q_loss = T.mean((Q - target) ** 2)

        nn.optimizers.Adam(a_loss + q_loss, lr)

        self.update = symjax.function(
            state,
            action,
            target,
            gradients,
            outputs=[a_loss, q_loss],
            updates=symjax.get_updates(),
        )
        g = symjax.gradients(T.mean(Q), [action])[0]
        self.get_gradients = symjax.function(state, action, outputs=g)

        # also create the target variants
        with symjax.Scope("actor_critic_target"):
            scaled_out_target = action_max * actor(state)
            Q_target = critic(state, action)

        self.actor_predict = symjax.function(state, outputs=scaled_out)
        self.actor_predict_target = symjax.function(state, outputs=scaled_out_target)
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

    def train(self):
        (
            s_batch,
            a_batch,
            r_batch,
            s2_batch,
            t_batch,
        ) = self.env.buffer.sample(self.batch_size)

        # Calculate targets
        target_q = self.critic_predict_target(
            s2_batch, self.actor_predict_target(s2_batch)
        )

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        discount = self.gamma ** self.reward.n
        y_i = r_batch[:, None] + discount * (1 - t_batch[:, None]) * target_q

        if not self.continuous:
            a_batch = (np.arange(self.num_actions) == a_batch[:, None]).astype(
                "float32"
            )

        td_error = np.abs(y_i - self.critic_predict(s_batch, a_batch))
        self.env.buffer.update_priorities(self.env.buffer.current_indices, td_error)

        # Update the critic given the targets
        gradients = self.get_gradients(s_batch, a_batch)
        a_loss, q_loss = self.update(s_batch, a_batch, y_i, gradients)

        # Update target networks
        self.update_target()
        return a_loss, q_loss


class OrnsteinUhlenbeckProcess:
    """dXt = theta*(mu-Xt)*dt + sigma*dWt"""

    def __init__(
        self,
        dim,
        theta=0.15,
        mu=0.0,
        sigma=0.2,
        noise_decay=0.99,
        initial_noise_scale=1,
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.noise_decay = noise_decay
        self.initial_noise_scale = initial_noise_scale
        self.reset()

    def __call__(self, action, episode):
        self.noise_process = self.theta * (
            self.mu - self.process
        ) + self.sigma * np.random.randn(self.dim)
        self.noise_scale = self.initial_noise_scale * self.noise_decay ** episode
        return action + self.noise_scale * self.process

    def reset(self):
        self.process = np.zeros(self.dim)


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

        (
            s_batch,
            a_batch,
            r_batch,
            s2_batch,
            t_batch,
            batch,
        ) = self.env.buffer.sample(self.batch_size)
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


def run(
    env,
    agent,
    reward,
    replay_size=int(1e6),
    action_noise=0.1,
    max_episode_length=500,
    decay=0.99,
    start_episode=100,
    update_after=10000,
    max_ep_steps=10000,
    max_episodes=1000,
    total_steps=40000,
    skip_frames=4,
):
    # Main loop: play episode and train

    init = 0.1 * (env.action_space.high - env.action_space.low)
    noise = OrnsteinUhlenbeckProcess(agent.num_actions, initial_noise_scale=init)
    global_step = 0
    losses = []
    for i in range(MAX_EPISODES):
        s = env.reset()

        # Clear episode buffer
        episode_buffer = []

        for j in range(max_ep_steps):

            a = noise(agent.act(s[None, :]), i)

            r = 0
            for k in range(skip_frames):
                s2, r_, terminal, info = env.step(a)
                r += r_
                if terminal:
                    break
            r /= k + 1

            episode_buffer.append([s, a, r, s2, terminal])
            s = s2

            # Perform the updates
            if env.buffer.len >= update_after:
                losses.append(agent.train())

            if terminal or j == (max_ep_steps - 1):

                episode_buffer = reward.discount(np.asarray(episode_buffer))
                print(
                    i,
                    " return:",
                    episode_buffer[:, 2].sum(),
                    "episode_length:",
                    j,
                    "noise scale",
                    noise.noise_scale,
                )

                for step in episode_buffer:
                    env.buffer.push(*step)

                print(np.array(losses).mean(0))
                noise.reset()
                break
    return (returns, q_losses, mu_losses)


def actor(state):
    input = nn.elu(nn.layers.Dense(state, 8))
    input = nn.elu(nn.layers.Dense(input, 8))
    input = nn.layers.Dense(input, 1)
    return T.tanh(input)


def critic(state, action):
    inputs = nn.layers.Dense(nn.elu(nn.layers.Dense(state, 8)), 8)
    inputa = nn.layers.Dense(nn.elu(nn.layers.Dense(action, 8)), 8)
    input = nn.elu(nn.layers.Dense(inputs + inputa, 8))
    input = nn.layers.Dense(input, 1)
    return input


# RL = DeepQNetwork(
#     env.n_actions,
#     env.n_states,
#     actor,
#     learning_rate=0.0005,
#     reward_decay=0.995,
#     e_greedy=0.1,
#     replace_target_iter=400,
#     batch_size=128,
#     memory_size=4000,
#     e_greedy_increment=None,
#     record_history=True,
# )

# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 1000
# Reward parameters
REWARD_FACTOR = 1  # Total episode reward factor
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 1e-2


# Size of replay buffer


reward = Reward(REWARD_FACTOR, GAMMA, n=1)

# env = pendulum.CartpoleEnv()
env = gym.make("Pendulum-v0")  # MountainCarContinuous-v0"
# )  # "Pendulum-v0")  # gym.make("CartPole-v0")

agent = ddpg(
    env,
    actor,
    critic,
    batch_size=128,
    tau=TAU,
    gamma=GAMMA,
    reward=reward,
)


env.buffer = BasicBuffer(size=int(1e5))


a, b, c = run(
    env,
    agent,
    reward,
    update_after=100,
    start_episode=100,
    skip_frames=1,
    max_episodes=10000,
    max_ep_steps=1000,
)

plt.subplot(211)
plt.plot(a)
plt.subplot(212)
plt.plot(b)
plt.plot(c)
plt.show()

# run_pendulum(env, RL)
