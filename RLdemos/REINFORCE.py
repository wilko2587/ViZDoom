import torch

from torch_models import SmallNet
import numpy as np
from torch import optim
from torch.distributions import Categorical

class agent:
    def __init__(self, actions):
        self.actions = actions

        self.network = SmallNet(
            in_height=160,
            in_width=128, # width needs 8 padding to fit architecture (divisible by 32)
            num_classes=len(actions),
        )

        self.gamma = 0.99
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=1e-4)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.terms = []

    def act(self, state: torch.FloatTensor):
        # calculate action and return the action
        logits = self.network(state)
        probs = torch.nn.functional.softmax(logits, dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def remember(self, tup):
        self.states.append(tup[0])
        self.actions.append(tup[1])
        self.log_probs.append(tup[2])
        self.rewards.append(tup[3])
        self.terms.append(tup[4])

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.terms = []

    def update_networks(self):
        # NB: we will assume the last element in each list is the terminal one

        # calculate the running rewards
#        Rs = [] # running rewards
#        R = 0 # initialise
#        for r in self.rewards[::-1]:
#            R = (r + R * self.gamma)/200
#            Rs.append(R)
#        Rs = Rs[::-1]

        # replay the episode and update the network
        reinforce_loss = 0 # initialise
        Rtot = sum(self.rewards)/100
        for t in range(len(self.states)):
            log_prob = self.log_probs[t]
            reinforce_loss += -log_prob * Rtot#Rs[t]

        self.optimizer.zero_grad()
        reinforce_loss.backward()
        self.optimizer.step()