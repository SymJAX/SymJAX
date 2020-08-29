# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
from . import agents

# https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44


# class DDPG(Agent):


class DDPG(agents.Agent):
    def __init__(
        self,
        state_shape,
        actions_shape,
        batch_size,
        actor,
        critic,
        lr=1e-3,
        gamma=0.99,
        tau=0.01,
    ):

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size

        states = T.Placeholder((batch_size,) + state_shape, "float32")
        actions = T.Placeholder((batch_size,) + actions_shape, "float32")
        self.critic = critic(states, actions)
        self.target_critic = critic(states, actions)

        # create critic loss
        targets = T.Placeholder(self.critic.q_values.shape.get(), "float32")
        critic_loss = ((self.critic.q_values - targets) ** 2).mean()

        # create optimizer
        with symjax.Scope("critic_optimizer"):
            nn.optimizers.Adam(critic_loss, lr, params=self.critic.params(True))

        # create the update function
        self._train_critic = symjax.function(
            states,
            actions,
            targets,
            outputs=critic_loss,
            updates=symjax.get_updates(scope="*/critic_optimizer"),
        )

        # now create utility function to get the gradients
        grad = symjax.gradients(self.critic.q_values.sum(), actions)
        self._get_critic_gradients = symjax.function(states, actions, outputs=grad)

        # create actor loss
        self.actor = actor(states)
        self.target_actor = actor(states)
        gradients = T.Placeholder(actions.shape.get(), "float32")
        actor_loss = -(self.actor.actions * gradients).mean()

        # create optimizer
        with symjax.Scope("actor_optimizer"):
            nn.optimizers.Adam(actor_loss, lr, params=self.actor.params(True))

        # create the update function
        self._train_actor = symjax.function(
            states,
            gradients,
            outputs=actor_loss,
            updates=symjax.get_updates(scope="*/actor_optimizer"),
        )

        # initialize both networks as the same
        self.update_target(1)

    def train(self, buffer, *args, **kwargs):

        s, a, r, s2, t = buffer.sample(self.batch_size)

        # Calculate the target for the critic
        a2 = self.target_actor.get_actions(s2)[0]
        q_values = self.target_critic.get_q_values(s2, a2)
        targets = r + (1 - t.astype("float32")) * self.gamma * q_values.squeeze()

        c_loss = self._train_critic(s, a, targets)

        # if not self.continuous:
        #     a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        actions = self.actor.get_actions(s)[0]
        gradients = self._get_critic_gradients(s, actions)

        a_loss = self._train_actor(s, gradients)
        self.update_target()
        return a_loss, c_loss


class REINFORCE(agents.Agent):
    """
    policy gradient reinforce also called reward-to-go policy gradient

    the vanilla policy gradient uses the total reward of each episode
    as a weight. In this implementation it is the discounted rewards to
    go that are used. Setting ``gamma`` to 1 leads to the reward to go
    policy gradient


    https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

    """

    def __init__(
        self,
        state_shape,
        actions_shape,
        n_episodes,
        episode_length,
        actor,
        lr=1e-3,
        gamma=0.99,
    ):
        self.actor = actor
        self.gamma = gamma
        self.lr = lr
        self.episode_length = episode_length
        self.n_episodes = n_episodes
        self.batch_size = episode_length * n_episodes

        states = T.Placeholder((self.batch_size,) + state_shape, "float32")
        actions = T.Placeholder((self.batch_size,) + actions_shape, "float32")
        discounted_rewards = T.Placeholder((self.batch_size,), "float32")

        self.actor = actor(states, distribution="gaussian")

        logprobs = self.actor.actions.log_prob(actions)
        actor_loss = -(logprobs * discounted_rewards).sum() / n_episodes

        with symjax.Scope("REINFORCE_optimizer"):
            nn.optimizers.Adam(
                actor_loss,
                lr,
                params=self.actor.params(True),
            )

        # create the update function
        self._train = symjax.function(
            states,
            actions,
            discounted_rewards,
            outputs=actor_loss,
            updates=symjax.get_updates(scope="*/REINFORCE_optimizer"),
        )

    def train(self, buffer, *args, **kwargs):

        assert buffer.n_episodes == self.n_episodes
        indices = list(range(self.episode_length * self.n_episodes))
        states, actions, disc_rewards = buffer.sample(
            indices,
            ["state", "action", "reward-to-go"],
        )

        disc_rewards -= disc_rewards.mean()
        disc_rewards /= disc_rewards.std()

        loss = self._train(states, actions, disc_rewards)
        buffer.reset()

        return loss


class ActorCritic(agents.Agent):
    """

    this corresponds to Q actor critic or V actor critic
    depending on the given critic

    (with GAE-Lambda for advantage estimation)

    https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/
    """

    def __init__(
        self,
        state_shape,
        actions_shape,
        n_episodes,
        episode_length,
        actor,
        critic,
        lr=1e-3,
        gamma=0.99,
        train_v_iters=10,
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lr = lr
        self.episode_length = episode_length
        self.n_episodes = n_episodes
        self.batch_size = episode_length * n_episodes
        self.train_v_iters = train_v_iters

        states = T.Placeholder((self.batch_size,) + state_shape, "float32")
        actions = T.Placeholder((self.batch_size,) + actions_shape, "float32")
        discounted_rewards = T.Placeholder((self.batch_size,), "float32")
        advantages = T.Placeholder((self.batch_size,), "float32")

        self.actor = actor(states, distribution="gaussian")
        self.critic = critic(states)

        logprobs = self.actor.actions.log_prob(actions)
        actor_loss = -(logprobs * advantages).sum() / n_episodes
        critic_loss = 0.5 * ((discounted_rewards - self.critic.q_values) ** 2).mean()

        with symjax.Scope("actor_optimizer"):
            nn.optimizers.Adam(
                actor_loss,
                lr,
                params=self.actor.params(True),
            )
        with symjax.Scope("critic_optimizer"):
            nn.optimizers.Adam(
                critic_loss,
                lr,
                params=self.critic.params(True),
            )

        # create the update function
        self._train_actor = symjax.function(
            states,
            actions,
            advantages,
            outputs=actor_loss,
            updates=symjax.get_updates(scope="*/actor_optimizer"),
        )
        # create the update function
        self._train_critic = symjax.function(
            states,
            discounted_rewards,
            outputs=critic_loss,
            updates=symjax.get_updates(scope="*/critic_optimizer"),
        )

    def train(self, buffer, *args, **kwargs):

        indices = list(range(self.batch_size))
        states, actions, disc_rewards, advantages = buffer.sample(
            indices,
            ["state", "action", "reward-to-go", "advantage"],
        )

        advantages -= advantages.mean()
        advantages /= advantages.std()

        actor_loss = self._train_actor(states, actions, advantages)
        for i in range(self.train_v_iters):
            critic_loss = self._train_critic(states, disc_rewards)
        buffer.reset()

        return actor_loss, critic_loss


class PPO(agents.Agent):
    """

    instead of using target networks one can record the old log probs

    have better advantage estimates

    """

    def __init__(
        self,
        state_shape,
        actions_shape,
        batch_size,
        actor,
        critic,
        lr=1e-3,
        K_epochs=80,
        eps_clip=0.2,
        gamma=0.99,
        entropy_beta=0.01,
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lr = lr
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        states = T.Placeholder((batch_size,) + state_shape, "float32", name="states")
        actions = T.Placeholder((batch_size,) + actions_shape, "float32", name="states")
        rewards = T.Placeholder((batch_size,), "float32", name="discounted_rewards")
        advantages = T.Placeholder((batch_size,), "float32", name="advantages")

        self.target_actor = actor(states, distribution="gaussian")
        self.actor = actor(states, distribution="gaussian")
        self.critic = critic(states)

        # Finding the ratio (pi_theta / pi_theta__old) and
        # surrogate Loss https://arxiv.org/pdf/1707.06347.pdf
        with symjax.Scope("policy_loss"):
            ratios = T.exp(
                self.actor.actions.log_prob(actions)
                - self.target_actor.actions.log_prob(actions)
            )
            ratios = T.clip(ratios, 0, 10)
            clipped_ratios = T.clip(ratios, 1 - self.eps_clip, 1 + self.eps_clip)

            surr1 = advantages * ratios
            surr2 = advantages * clipped_ratios

            actor_loss = -(T.minimum(surr1, surr2)).mean()

        with symjax.Scope("monitor"):
            clipfrac = (
                ((ratios > (1 + self.eps_clip)) | (ratios < (1 - self.eps_clip)))
                .astype("float32")
                .mean()
            )
            approx_kl = (
                self.target_actor.actions.log_prob(actions)
                - self.actor.actions.log_prob(actions)
            ).mean()

        with symjax.Scope("critic_loss"):
            critic_loss = T.mean((rewards - self.critic.q_values) ** 2)

        with symjax.Scope("entropy"):
            entropy = self.actor.actions.entropy().mean()

        loss = actor_loss + critic_loss  # - entropy_beta * entropy

        with symjax.Scope("optimizer"):
            nn.optimizers.Adam(
                loss,
                lr,
                params=self.actor.params(True) + self.critic.params(True),
            )

        # create the update function
        self._train = symjax.function(
            states,
            actions,
            rewards,
            advantages,
            outputs=[actor_loss, critic_loss, clipfrac, approx_kl],
            updates=symjax.get_updates(scope="*optimizer"),
        )

        # initialize target as current
        self.update_target(1)

    def train(self, buffer, *args, **kwargs):

        indices = list(range(buffer.length))
        states, actions, rewards, advantages = buffer.sample(
            indices,
            ["state", "action", "reward-to-go", "advantage"],
        )

        # Optimize policy for K epochs:
        advantages -= advantages.mean()
        advantages /= advantages.std()

        for _ in range(self.K_epochs):

            for s, a, r, adv in symjax.data.utils.batchify(
                states,
                actions,
                rewards,
                advantages,
                batch_size=self.batch_size,
            ):

                loss = self._train(s, a, r, adv)

        print([v.value for v in symjax.get_variables(name="logsigma")])
        buffer.reset_data()
        self.update_target(1)
        return loss
