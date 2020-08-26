# main class

import symjax
import numpy as np
from symjax import nn
import symjax.tensor as T
from . import agents


# https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44


# class DDPG(Agent):


class DDPG(agents.Agent):
    def __init__(self, actor, critic, lr=1e-3, gamma=0.99):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lr = lr

        # create critic loss
        targets = T.Placeholder(critic.q_values.shape.get(), "float32")
        critic_loss = ((self.critic.q_values - targets) ** 2).mean()

        # create optimizer
        with symjax.Scope("critic_optimizer"):
            nn.optimizers.Adam(critic_loss, lr, params=self.critic.params)

        # create the update function
        self._train_critic = symjax.function(
            self.critic.states,
            self.critic.actions,
            targets,
            outputs=critic_loss,
            updates=symjax.get_updates(scope="*critic_optimizer"),
        )

        # create actor loss
        gradients = T.Placeholder(self.actor.actions.shape.get(), "float32")
        actor_loss = -(gradients * self.actor.actions).mean()

        # create optimizer
        with symjax.Scope("actor_optimizer"):
            nn.optimizers.Adam(actor_loss, lr, params=self.actor.params)

        # create the update function
        self._train_actor = symjax.function(
            self.actor.states,
            gradients,
            outputs=actor_loss,
            updates=symjax.get_updates(scope="*actor_optimizer"),
        )

        # now create utility function to get the gradients
        grad = symjax.gradients(self.critic.q_values.sum(), self.critic.actions)
        self._get_critic_gradients = symjax.function(
            self.critic.states, self.critic.actions, outputs=grad
        )

    def train(self, buffer, *args, **kwargs):

        s, a, r, s2, t = buffer.sample(self.batch_size)

        # Calculate the target for the critic
        a2 = self.actor.get_target_actions(s2)
        q_values = self.critic.get_target_q_values(s2, a2)
        targets = r + (1 - t.astype("float32")) * self.gamma * q_values.squeeze()

        c_loss = self._train_critic(s, a, targets[:, None])

        # if not self.continuous:
        #     a = (np.arange(self.num_actions) == a[:, None]).astype("float32")

        actions = self.actor.get_actions(s)
        gradients = self._get_critic_gradients(s, actions)

        a_loss = self._train_actor(s, gradients)
        # print(a_loss, c_loss, actions[:4])
        self.actor.update_target()
        self.critic.update_target()
        return 0, 0
