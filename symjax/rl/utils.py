from collections import deque
import random
from scipy.signal import lfilter
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

    normalize: bool
        advantage normalization trick

    """

    def __init__(self, factor, gamma=None, n=8, normalize=False, eps=1e-6):
        # Reward parameters
        self.factor = factor
        self.n = n
        self.gamma = gamma
        self.normalize = normalize
        self.eps = eps
        self.ptr, self.path_start_idx = 0, 0

    # Set step rewards to total episode reward
    def total(self, ep_batch, tot_reward):
        for step in ep_batch:
            step[2] = tot_reward * self.factor
        return ep_batch

    # Set step rewards to discounted reward
    def discount(self, ep_batch):

        if self.n == np.inf:
            rewards = []
            for reward, is_terminal in zip(
                reversed(ep_batch[:, 2]), reversed(ep_batch[:, 4])
            ):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            ep_batch[:, 2] = np.asarray(rewards)

        elif self.n != 1:
            x = ep_batch[:, 2]

            if self.n == np.inf:
                b = [1]
                a = [1, -self.gamma]
            else:
                b = self.gamma ** np.arange(self.n)
                a = [1]
            ep_batch[:, 2] = lfilter(b=b, a=a, x=x[::-1], axis=0)[::-1]

        # Normalize
        if self.normalize:
            std = np.std(ep_batch[:, 2]) + self.eps
            ep_batch[:, 2] = (ep_batch[:, 2] - np.mean(ep_batch[:, 2])) / std

        ep_batch[:, 2] *= self.factor

        return ep_batch


def discount_cumsum(x, discount):
    """
    computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """

    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer(dict):
    """
    Bufer holding different values of experience

    By default this contains ``"reward", "reward-to-go", "V" or "Q", "action", "state", "episode", "priorities", "TD-error", "terminal", "next-state"``
    𝑄𝜋(𝑠,𝑎)=𝐸𝜋{𝑅𝑡|𝑠𝑡=𝑠,𝑎𝑡=𝑎}=𝐸𝜋{∑𝑘=0∞𝛾𝑘𝑟𝑡+𝑘+1|𝑠𝑡=𝑠,𝑎𝑡=𝑎}
    𝑉𝜋(𝑠)=𝐸𝜋{𝑅𝑡|𝑠𝑡=𝑠}=𝐸𝜋{∑𝑘=0∞𝛾𝑘𝑟𝑡+𝑘+1|𝑠𝑡=𝑠}
    𝛾∈[0,1] is called discount factor and determines if one focuses on immediate rewards (𝛾=0), the total reward (𝛾=1) or some trade-off.
    lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
    """

    def __init__(
        self,
        action_shape,
        state_shape,
        size,
        V_or_Q,
        extras=None,
        priority_sampling=False,
        gamma=0.99,
        lam=0.97,
    ):

        self.priority_sampling = priority_sampling
        self.size = size
        self.lam = lam
        self.episode_length = 0
        self.gamma = gamma
        self.extras = {} if extras is None else extras
        self.action_shape = action_shape
        self.state_shape = state_shape

        self.V_or_Q = V_or_Q
        assert V_or_Q in ["V", "Q"]
        super().__init__()
        self.reset()

    def reset(self):
        for item in list(self.keys()):
            self.pop(item)

        self["reward"] = np.empty((self.size,))
        self["reward-to-go"] = np.empty((self.size,))
        self["advantage"] = np.empty((self.size,))
        self["terminal"] = np.empty((self.size,), dtype="bool")
        self["TD-error"] = np.empty((self.size,))
        if self.V_or_Q == "V":
            self[self.V_or_Q] = np.empty(self.size)
        else:
            self[self.V_or_Q] = np.empty(self.size + self.action_shape)
        self["action"] = np.empty((self.size,) + self.action_shape)
        self["state"] = np.empty((self.size,) + self.state_shape)
        self["next-state"] = np.empty((self.size,) + self.state_shape)
        self["episode"] = np.zeros((self.size,), dtype="int32") - 1
        self["priority"] = np.ones((self.size,))

        for key, shape in self.extras.items():
            self[key] = np.empty((self.size,) + shape)

        self.ptr = 0
        self.path_start_ptr = 0
        self.episode_reward = 0
        self._length = 0

    def push(self, kwargs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert "episode" in kwargs
        assert "action" in kwargs
        assert "reward" in kwargs
        assert "state" in kwargs
        assert "next-state" in kwargs
        assert "terminal" in kwargs
        assert self.V_or_Q in kwargs

        for key, value in kwargs.items():
            self[key][self.ptr % self.size] = value

        self.ptr += 1
        self.episode_reward += kwargs["reward"]
        self._length += 1

    @property
    def length(self):
        return max(self._length, self.size)

    def sample(self, length, keys=None):

        if keys is None:
            keys = ["state", "action", "reward", "next-state", "terminal"]

        valid = self["episode"] >= 0
        indices = np.arange(self.size, dtype="int32")[valid]
        if self.priority_sampling:
            p = np.array(self.priorities)[valid]
            p /= p.sum()
            batch = np.random.choice(indices, p=p, size=length)
        else:
            batch = np.random.choice(indices, size=length)

        batch = indices[:length]
        self.current_indices = batch

        outputs = []
        for key in keys:
            outputs.append(self[key][batch])

        return outputs

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = np.arange(self.path_start_ptr, self.ptr) % self.size

        # the next lines implement GAE-Lambda advantage calculation
        if "V" not in self:
            assert "Q" in self
            # 𝑄𝜋(𝑠,𝜋(𝑠))=𝑉𝜋(𝑠)
            vals = np.append(self["Q"][path_slice], last_val).sum(1)
        else:
            vals = np.append(self["V"][path_slice], last_val)
        rews = np.append(self["reward"][path_slice], last_val)

        # temporal-difference (TD) error
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self["TD-error"][path_slice] = deltas

        self["advantage"][path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # computes rewards-to-go, (targets of value function)
        self["reward-to-go"][path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self["advantage"][path_slice] = self["reward-to-go"][path_slice] - vals[:-1]

        self["priority"][path_slice] = 1 + deltas

        self.path_start_ptr = self.ptr
        self.episode_reward = 0


class OrnsteinUhlenbeckProcess:
    """dXt = theta*(mu-Xt)*dt + sigma*dWt"""

    def __init__(
        self,
        mean=0.0,
        std_dev=0.2,
        theta=0.15,
        dt=1e-2,
        noise_decay=0.99,
        initial_noise_scale=1,
        init=None,
    ):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = (dt,)
        self.init = init
        self.noise_decay = noise_decay
        self.initial_noise_scale = initial_noise_scale
        self.end_episode()

    def __call__(self, action, episode):

        self.noise_scale = self.initial_noise_scale * self.noise_decay ** episode

        x = (
            self.process
            + self.theta * (self.mean - self.process) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=action.shape)
        )
        # Store x into process
        # Makes next noise dependent on current one
        self.process = x

        return action + self.noise_scale * self.process

    def end_episode(self):
        if self.init is None:
            self.process = np.zeros(1)
        else:
            self.process = self.init


class Gaussian:
    """dXt = theta*(mu-Xt)*dt + sigma*dWt"""

    def __init__(
        self, dim, mu=0.0, sigma=1, noise_decay=0.99, initial_noise_scale=1,
    ):
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.noise_decay = noise_decay
        self.initial_noise_scale = initial_noise_scale

    def __call__(self, action, episode):
        noise = np.random.randn(self.dim) * self.sigma + self.mu
        self.noise_scale = self.initial_noise_scale * self.noise_decay ** episode
        return action + self.noise_scale * noise

    def end_episode(self):
        pass


def run(
    env,
    agent,
    buffer,
    rewarder=None,
    noise=None,
    action_processor=None,
    max_episode_steps=10000,
    max_episodes=1000,
    update_every=1,
    update_after=1,
    skip_frames=1,
    reset_each_episode=False,
    wait_end_path=False,
):
    global_step = 0
    losses = []
    for i in range(max_episodes):
        state = env.reset()

        for j in range(max_episode_steps):

            action, extra = agent.act(state[None, :])
            global_step += 1

            if noise:
                action = noise(action, episode=i)
            if action_processor:
                action = action_processor(action)

            reward = 0
            for k in range(skip_frames):
                next_state, r, terminal, info = env.step(action)
                reward += r
                if terminal:
                    break
            reward /= k + 1

            base = {
                "state": state,
                "action": action,
                "reward": reward,
                "next-state": next_state,
                "terminal": terminal,
                "episode": i,
            }
            base.update(extra)
            buffer.push(base)

            state = next_state

            # Perform the updates
            if (
                buffer.length >= agent.batch_size
                and global_step > update_after
                and global_step % update_every == 0
                and not wait_end_path
            ):
                losses.append(agent.train(buffer, episode=i, step=j))

            if terminal or j == (max_episode_steps - 1):

                print(
                    "Episode:",
                    i,
                    ", return:",
                    buffer.episode_reward,
                    "episode_length:",
                    j,
                )

                # bootstrap value
                if not terminal:
                    buffer.finish_path(agent.get_v(state))
                else:
                    buffer.finish_path(0)

                if wait_end_path and global_step > update_after:
                    losses.append(agent.train(buffer, episode=i, step=j))

                if noise:
                    noise.end_episode()

                break
        if reset_each_episode:
            buffer.reset()
    return (returns, q_losses, mu_losses)
