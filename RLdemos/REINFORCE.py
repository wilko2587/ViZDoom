import torch
from torch_models import SmallNet
import os
from torch import optim
from torch.distributions import Categorical

class agent:
    def __init__(self, n_actions,
                 gradient_accumulation: int = 1,
                 lr: float = 1e-4):

        self.n_actions = n_actions

        self.network = SmallNet(
            in_height=160,
            in_width=128, # width needs 8 padding to fit architecture (divisible by 32)
            num_classes=n_actions,
        )

        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr)
        self.states = []
        self.n_actions = []
        self.log_probs = []
        self.rewards = []
        self.terms = []

        self.gradient_accumulation = gradient_accumulation
        self.gradient_accumulation_counter = 0

    def act(self, state: torch.FloatTensor):
        # calculate action and return the action
        logits = self.network(state)
        probs = torch.nn.functional.softmax(logits, dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def remember(self, tup):
        self.states.append(tup[0])
        self.n_actions.append(tup[1])
        self.log_probs.append(tup[2])
        self.rewards.append(tup[3])
        self.terms.append(tup[4])

    def reset(self):
        self.states = []
        self.n_actions = []
        self.log_probs = []
        self.rewards = []
        self.terms = []

    def save_model(self, name):
        if name[-3:] != '.pt':
            name = name + '.pt'
        torch.save(self.network.state_dict(), os.path.join('TorchModels/', name))

    def load_model(self, name):
        x = torch.load(os.path.join('./TorchModels/', name))
        self.network.load_state_dict(x)

    def update_networks(self):

        # replay the episode and update the network
        reinforce_loss = 0 # initialise
        Rtot = sum(self.rewards)/100
        for t in range(len(self.states)):
            log_prob = self.log_probs[t]
            reinforce_loss += -log_prob * Rtot

        reinforce_loss.backward()
        self.gradient_accumulation_counter += 1
        # we can update the network with accumulation of self.gradient_accumulation
        # episodes to improve learning stability

        if self.gradient_accumulation_counter % self.gradient_accumulation == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.gradient_accumulation_counter = 0 # reset