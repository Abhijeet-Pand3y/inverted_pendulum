"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

from Modules import NormalModule


class Buffer:
    """Experience replay buffer storing one-step transitions.

    Use-contract:
        add(state, action, reward, done)             — push one transition
        calc_reward_to_go(gamma)                     — fill self.ret_to_go
        sample(batch_size) -> tuple of numpy arrays  — draw a mini-batch
    """

    def __init__(self, sdim, adim, size, sdtype=np.float32, adtype=np.float32, ep_len=200):
        self.states    = np.zeros((size, sdim), dtype=sdtype)
        self.actions   = np.zeros((size, adim), dtype=adtype)
        self.rewards   = np.zeros((size, 1),    dtype=np.float32)
        self.ret_to_go = np.zeros((size, 1),    dtype=np.float32)
        self.dones     = np.zeros((size, 1),    dtype=bool)
        self.i     = 0
        self.size  = size
        self.max_i = 0
        self.ep_len = ep_len

    def add(self, state, action, reward, done):
        self.states[self.i]    = state
        self.actions[self.i]   = action
        self.rewards[self.i]   = reward
        self.dones[self.i]     = done
        self.i                 = (self.i + 1) % self.size
        self.max_i             = min(self.max_i + 1, self.size)


    def sample(self, batch_size):
        upper = max(self.max_i - 1, 1)
        idxs = np.random.randint(0, upper, size=batch_size)
        done_mask = self.dones[idxs, 0]
        idxs = np.where(done_mask, np.maximum(idxs - 1, 0), idxs)
        next_idxs = idxs + 1
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[next_idxs],
            self.dones[idxs],
            self.ret_to_go[idxs],
            self.ret_to_go[next_idxs],
        )

    def calc_reward_to_go(self, gamma=0.975):
        running_return = 0.0

        for i in range(self.max_i - 1, -1, -1):
            if self.dones[i, 0]:
                running_return = 0.0

            running_return = self.rewards[i, 0] + gamma * running_return
            self.ret_to_go[i, 0] = running_return



def collect_data(size, env, agent, title="collecting"):
    """Roll out `agent` (a policy network) in `env` for `size` steps.

    Returns:
        buffer  — a populated Buffer
        avg_rwd — average per-step reward observed during the rollout
    """
    state, _ = env.reset()       # state shape gives you sdim
    sdim = state.shape[0]
    adim = env.action_space.sample().shape[0]
    amin = env.action_space.low[0]
    amax = env.action_space.high[0]
    buffer = Buffer(sdim=sdim, adim=adim, size=size)
    total_reward = 0
    for i in range(size):
        action = act(agent, state, amin, amax)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, done)
        total_reward += reward
        if done:
            state, _ = env.reset()
        else:
            state = next_state

    return buffer, total_reward/size



def act(policy, state, amin, amax):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    with th.no_grad():
        state_tensor = th.as_tensor(state, dtype=th.float32).unsqueeze(0)
        mu, sigma = policy(state_tensor)
        action = Normal(mu, sigma).sample()

        action = th.tanh(action)
        rescaled_action = rescale_actions(action, amin, amax)

    return rescaled_action.flatten().cpu().numpy()
    


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    rescaled_action = amin + ((action + 1)/2) * (amax-amin)# Rescaling from [-1,1] to [-2,2] for tanh
    return rescaled_action