from collections import deque
import random
from scipy.signal import lfilter
import numpy as np
from collections import deque


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
    Buffer holding different values of experience

    By default this contains ``"reward", "reward-to-go", "V" or "Q", "action", "state", "episode", "priorities", "TD-error", "terminal", "next-state"``
    𝑄𝜋(𝑠,𝑎)=𝐸𝜋{𝑅𝑡|𝑠𝑡=𝑠,𝑎𝑡=𝑎}=𝐸𝜋{∑𝑘=0∞𝛾𝑘𝑟𝑡+𝑘+1|𝑠𝑡=𝑠,𝑎𝑡=𝑎}
    𝑉𝜋(𝑠)=𝐸𝜋{𝑅𝑡|𝑠𝑡=𝑠}=𝐸𝜋{∑𝑘=0∞𝛾𝑘𝑟𝑡+𝑘+1|𝑠𝑡=𝑠}
    𝛾∈[0,1] is called discount factor and determines if one focuses on immediate rewards (𝛾=0), the total reward (𝛾=1) or some trade-off.
    lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
    """

    def __init__(
        self,
        maxlen,
        priority_sampling=False,
        gamma=0.99,
        lam=0.95,
    ):

        self.priority_sampling = priority_sampling
        self.maxlen = maxlen
        self.lam = lam
        self.gamma = gamma

        super().__init__()
        self.reset()

    def reset_data(self):
        for item in list(self.keys()):
            self.pop(item)
        self._length = 0

    def reset(self):
        self.reset_data()

        self.episode_reward = 0
        self.episode_length = 0
        self.n_episodes = 0

        # runing stats
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
        self.mean = np.zeros((), "float64")
        self.var = np.ones((), "float64")
        self.std = np.ones((), "float64")
        self.count = 1e-4

    def push(self, kwargs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        for key, value in kwargs.items():
            if key not in self:
                self[key] = deque(maxlen=self.maxlen)
            self[key].append(value)

        if "reward" in kwargs:
            self.episode_reward += kwargs["reward"]
            self.episode_length += 1
            self._length += 1

    @property
    def length(self):
        return min(self._length, self.maxlen)

    def sample(self, length_or_indices, keys=None):

        if keys is None:
            keys = ["state", "action", "reward", "next-state", "terminal"]

        if np.isscalar(length_or_indices):
            indices = np.arange(self.length, dtype="int32")
            if self.priority_sampling:
                p = np.array(self.priorities)[valid]
                p /= p.sum()
                batch = np.random.choice(indices, p=p, size=length_or_indices)
            else:
                batch = np.random.choice(indices, size=length_or_indices)

            self.current_indices = batch
        else:
            self.current_indices = length_or_indices

        outputs = []
        for key in keys:
            outputs.append(np.array([self[key][i] for i in self.current_indices]))

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
        should be :math:`V(s_T)`, the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # check how far in the past (in the episode) can we get data
        back = min(self.episode_length, self.length)

        # create the indices to conveniently access those data though
        # the sample method
        indices = range(-back, 0)

        if "reward" in self:

            # access episode rewards
            rews = np.append(self.sample(indices, ["reward"])[0], last_val)
            # update running stats
            self.update_stats(rews)
            rews = np.clip((rews - self.mean) / self.std, -10, 10)

            # computes rewards-to-go, (targets of value function) and push them
            for val in discount_cumsum(rews, self.gamma)[:-1]:
                self.push({"reward-to-go": val})

        if "V" in self:

            # access state-values of the episode
            vals = np.append(
                self.sample(indices, ["V"])[0],
                last_val,
            )

            # computes GAE-Lambda advantage calculation
            # https://arxiv.org/abs/1506.02438
            advantages = generalized_advantage_estimation(
                vals, rews, self.gamma * self.lam
            )
            for v in advantages:
                self.push({"advantage": v})

            # # also add the returns
            # for v in advantages + vals[:-1]:
            #     self.push({"return": v})

            # self["priority"][path_slice] = (
            #     1 + rews[:-1] - vals[:-1] + self.gamma * vals[1:]
            # )

        self.episode_reward = 0
        self.episode_length = 0
        self.n_episodes += 1

    def update_stats(self, x):

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count


def generalized_advantage_estimation(values, rewards, gamma, lam=0.95):
    """
    GAE-Lambda advantage calculation

    Args:

    values: the V_hat(s)

    reward:
        as return by the environment

    gamma: reward discount

    lam: lambda (see paper).
        lam=0 : use TD residuals
        lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    """

    # temporal-difference (TD) error
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = discount_cumsum(deltas, gamma * lam)
    return advantages


class Gaussian:
    """dXt = theta*(mu-Xt)*dt + sigma*dWt"""

    def __init__(
        self,
        dim,
        mu=0.0,
        sigma=1,
        noise_decay=0.99,
        initial_noise_scale=1,
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
    eval_every=10,
    eval_max_episode_steps=10000,
    eval_max_episodes=10,
):
    global_step = 0
    all_losses = []
    all_train_rewards = []
    all_evals = []

    for episode in range(max_episodes):
        state = env.reset()
        all_losses.append([])

        for j in range(max_episode_steps):
            # action = agent.get_noise_action(state)
            state, terminal, infos = agent.play(state, env, skip_frames=skip_frames)
            infos.update({"episode": episode})
            buffer.push(infos)

            global_step += 1

            # Perform the updates
            if (
                buffer.length >= agent.batch_size
                and global_step > update_after
                and global_step % update_every == 0
                and not wait_end_path
            ):
                all_losses[-1].append(agent.train(buffer, episode=episode, step=j))

            if terminal or j == (max_episode_steps - 1):

                # bootstrap value
                # if not terminal:
                #     buffer.finish_path(agent.get_v(state))
                # else:
                all_train_rewards.append(buffer.episode_reward)
                print(
                    "Episode: {}\n\
                        return: {}\n\
                        episode_length: {}\n\
                        losses: {}".format(
                        episode, buffer.episode_reward, j, all_losses[-1][-1:]
                    )
                )
                buffer.finish_path(0)

                if (
                    wait_end_path
                    and global_step >= update_after
                    and global_step % update_every == 0
                    and buffer.length >= agent.batch_size
                ):
                    all_losses[-1].append(agent.train(buffer, episode=episode, step=j))
                    print(all_losses[-1][-1:])

                if noise:
                    noise.end_episode()

                break
        if reset_each_episode:
            buffer.reset()

        if episode % eval_every == 0:
            all_evals.append(
                play(env, agent, eval_max_episodes, eval_max_episode_steps)
            )
            ave_reward = all_evals[-1][0].sum() / all_evals[-1][1].sum()
            print("episode: ", episode, "Evaluation Average Reward:", ave_reward)

    return all_losses, all_train_rewards, all_evals


def play(environment, agent, max_episodes, max_episode_steps):
    episode_rewards = []
    episode_steps = []
    for i in range(max_episodes):
        episode_rewards.append(0)
        episode_steps.append(0)
        state = environment.reset()
        for j in range(max_episode_steps):
            action = agent.get_action(state)[0]
            state, reward, done, _ = environment.step(action)
            episode_rewards[-1] += reward
            episode_steps[-1] += 1
            if done:
                break
    return np.array(episode_rewards), np.array(episode_steps)
