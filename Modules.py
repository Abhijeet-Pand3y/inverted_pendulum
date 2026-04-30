"""Utility modules provided for the assignment.

`NormalModule` is the final head of the policy network: it takes a feature
vector and returns (mu, sigma) parameterising a Gaussian action distribution.
The mean is tanh-squashed into (-1, 1); you will need to rescale it into the
env's action range before stepping the environment. The log standard
deviation is a learnable but state-independent parameter.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalModule(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        # mu is squashed to (-1, 1); rescale it to the env action range later.
        return F.tanh(mout), vout
