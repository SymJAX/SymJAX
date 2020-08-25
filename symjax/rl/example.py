# main class

from symjax import nn
import symjax.tensor as T
import symjax
import matplotlib.pyplot as plt
import gym
import utils
import agents
from ddpg import DDPG
from gym import wrappers


def init_b(shape):
    return nn.initializers.uniform(shape, 0.05)


class actor(agents.Actor):
    def create_network(self, state):
        input = nn.relu(nn.layers.Dense(state, 32))
        input = nn.relu(nn.layers.Dense(input, 32))
        input = nn.layers.Dense(input, 1)
        return input


class critic(agents.Critic):
    def create_network(self, state, action):
        input = nn.relu(nn.layers.Dense(T.concatenate([state, action], 1), 300))
        input = nn.relu(nn.layers.Dense(input, 300))
        input = nn.layers.Dense(input, 1)
        return input


# environment
outdir = "/tmp/DDPG-agent-results"
env = gym.make("Pendulum-v0")
env = wrappers.Monitor(env, outdir, force=True)
# MountainCarContinuous-v0"
# )  # "Pendulum-v0")  # gym.make("CartPole-v0")

# reward smoothing
rewarder = utils.NStepRewarder(factor=1, gamma=0.99, n=1)

# agent
# agent = ddpg(env, actor, critic, batch_size=64, tau=0.01, gamma=0.99, lr=1e-4)

# buffer
buffer = utils.Buffer(action_shape=(1,), state_shape=(3,), V_or_Q="V", size=int(1e6))

# noise exploration
init = 0.1 * (env.action_space.high - env.action_space.low)
# noise = utils.OrnsteinUhlenbeckProcess(
#     dim=agent.num_actions, initial_noise_scale=init
# )

agent = DDPG(
    actor(batch_size=64, state_shape=(3,)),
    critic(batch_size=64, state_shape=(3,), action_shape=(1,)),
)

utils.run(
    env,
    agent,
    buffer,
    rewarder=None,
    noise=None,
    skip_frames=1,
    update_after=10000,
    wait_end_path=False,
    max_episodes=20,
)

symjax.save_variables("saved_model")
symjax.load_variables("saved_model")
# run_pendulum(env, RL)
