#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

HIDDEN_SIZE = 64

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.fc_adv = nn.Sequential(
            nn.Linear(input_shape[0], HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(input_shape[0], HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean()
