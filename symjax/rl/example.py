# main class

from symjax import nn
import symjax.tensor as T
import matplotlib.pyplot as plt
import gym
import utils
from ddpg import ddpg


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


# environment
env = gym.make("Pendulum-v0")  # MountainCarContinuous-v0"
# )  # "Pendulum-v0")  # gym.make("CartPole-v0")

# reward smoothing
rewarder = utils.NStepRewarder(factor=1, gamma=0.99, n=1)

# agent
agent = ddpg(env, actor, critic, batch_size=128, tau=0.01, gamma=0.99)

# buffer
buffer = utils.Buffer(size=int(1e5))

# noise exploration
init = 0.1 * (env.action_space.high - env.action_space.low)
noise = utils.OrnsteinUhlenbeckProcess(dim=agent.num_actions, initial_noise_scale=init)


utils.run(env, agent, buffer, rewarder=rewarder, noise=noise, skip_frames=1)

plt.subplot(211)
plt.plot(a)
plt.subplot(212)
plt.plot(b)
plt.plot(c)
plt.show()

# run_pendulum(env, RL)
