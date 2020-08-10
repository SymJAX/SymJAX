from collections import deque
import random
import scipy.signal
import numpy as np


class NStepRewarder(object):
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


class Buffer:
    def __init__(self, size, with_priorities=False):

        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.with_priorities = with_priorities

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
        self.end_episode()

    def __call__(self, action, episode):
        self.noise_process = self.theta * (
            self.mu - self.process
        ) + self.sigma * np.random.randn(self.dim)
        self.noise_scale = self.initial_noise_scale * self.noise_decay ** episode
        return action + self.noise_scale * self.process

    def end_episode(self):
        self.process = np.zeros(self.dim)


def run(
    env,
    agent,
    buffer,
    rewarder=None,
    noise=None,
    max_episode_steps=10000,
    max_episodes=1000,
    skip_frames=1,
):
    global_step = 0
    losses = []
    for i in range(max_episodes):
        state = env.reset()

        # Clear episode buffer
        episode_buffer = []

        for j in range(max_episode_steps):

            if noise:
                action = noise(agent.act(state[None, :]), i)
            else:
                action = agent.act(state[None, :])

            reward = 0
            for k in range(skip_frames):
                next_state, r, terminal, info = env.step(action)
                reward += r
                if terminal:
                    break
            reward /= k + 1

            episode_buffer.append([state, action, reward, next_state, terminal])
            state = next_state

            # Perform the updates
            if buffer.len >= agent.batch_size:
                losses.append(agent.train(buffer, episode=i, step=j))

            if terminal or j == (max_episode_steps - 1):

                if rewarder:
                    episode_buffer = rewarder.discount(np.asarray(episode_buffer))

                for step in episode_buffer:
                    buffer.push(*step)

                if noise:
                    noise.end_episode()

                print(
                    "Episode:",
                    i,
                    ", return:",
                    episode_buffer[:, 2].sum(),
                    "episode_length:",
                    j,
                )

                break
    return (returns, q_losses, mu_losses)
