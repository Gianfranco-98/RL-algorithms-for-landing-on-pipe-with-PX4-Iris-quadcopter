#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

HIDDEN_SIZE = 128

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape[0], HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions)
        )

    def forward(self, x):
        return self.net(x)
