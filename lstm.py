
import numpy as np
import torch as th
import torch.nn as nn
from Modules import NormalModule



class RecurrentActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.head = NormalModule(hidden_size, action_dim)
    
    def forward(self, state, hidden):
        # state shape: (batch, seq_len, state_dim)
        # hidden: (h, c) each shape (1, batch, hidden_size)
        out, new_hidden = self.lstm(state, hidden)
        mu, sigma = self.head(out[:, -1, :])  # take last timestep
        return mu, sigma, new_hidden