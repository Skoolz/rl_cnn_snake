import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.fc_adv1 = nn.Linear(576, 512)
        self.fc_adv2 = nn.Linear(512, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        #return adv
        return F.softmax(adv,1)
    
    def act(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
