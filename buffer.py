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
        self.next_states = np.zeros((size, sdim), dtype=sdtype)
        self.i     = 0
        self.size  = size
        self.max_i = 0
        self.ep_len = ep_len

    def add(self, state, action, reward, done, next_state):
        self.states[self.i]    = state
        self.actions[self.i]   = action
        self.rewards[self.i]   = reward
        self.next_states[self.i] = next_state
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
            self.dones[next_idxs],
            self.ret_to_go[idxs],
            self.ret_to_go[next_idxs],
        )

    def calc_reward_to_go(self, gamma=0.975):
        for i in range(self.max_i-1,-1,-1):
            if self.dones[i] or i == self.max_i-1:
                self.ret_to_go[i] = self.rewards[i]
            else:
                self.ret_to_go[i] = self.rewards[i] + gamma * self.ret_to_go[i+1] 
    
    def calc_reward_to_go_parallel(self, gamma=0.975):
        for i in range(self.max_i-1,-1,-1):
            if self.max_i-1 == i:
                self.ret_to_go[i] = self.rewards[i]
            else:
                non_terminal_mask = 1.0 - self.dones[i]
                self.ret_to_go[i] = self.rewards[i] + gamma * self.ret_to_go[i+1] * non_terminal_mask



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
        buffer.add(state, action, reward, done, next_state)
        total_reward += reward
        if done:
            state, _ = env.reset()
        else:
            state = next_state

    return buffer, total_reward/size


def collect_data_parallel(steps_per_envs, envs, num_envs, agents, title="collecting parallel"):
    buffer_size = steps_per_envs * num_envs

    states, _ = envs.reset()
    sdim = states.shape[1]
    adim = envs.single_action_space.sample().shape[0]
    amin = envs.single_action_space.low[0]
    amax = envs.single_action_space.high[0]
    buffer = Buffer(sdim, adim, buffer_size)
    total_reward = 0
    for i in range(steps_per_envs):
        actions = act_parallel(agents, states, amin, amax)
        next_states, rewards, terminated, truncated, info = envs.step(actions)
        dones = terminated | truncated
        for j in range(num_envs):
            buffer.add(states[j], actions[j], rewards[j], dones[j], next_states[j])
        total_reward += rewards.sum()
        states = next_states
    
    return buffer, total_reward/(buffer_size)


def act(policy, state, amin, amax):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    state_tensor = th.as_tensor(state, dtype=th.float32).unsqueeze(0)
    mu, sigma = policy(state_tensor)
    action = Normal(mu, sigma).sample()
    action = th.clamp(action, -1, 1)

    rescaled_action = rescale_actions(action, amin, amax)

    return np.array(rescaled_action.flatten())

def act_parallel(policy, states, amin, amax):
    states_tensor = th.as_tensor(states, dtype=th.float32)
    mu, sigma = policy(states_tensor)
    actions = Normal(mu, sigma).sample()
    actions = th.clamp(actions, -1, 1)
    return rescale_actions(actions, amin, amax).detach().numpy()


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    rescaled_action = amin + ((action + 1)/2) * (amax-amin)# Rescaling from [-1,1] to [-2,2] for tanh
    return rescaled_action

def act_recurrent(policy, state, hidden, amin, amax):
    state_tensor = th.as_tensor(state, dtype=th.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
    with th.no_grad():
        mu, sigma, new_hidden = policy(state_tensor, hidden)
    action = Normal(mu, sigma).sample()
    action = th.clamp(action, -1, 1)
    return rescale_actions(action, amin, amax).detach().numpy().flatten(), new_hidden

def collect_data_recurrent(size, env, policy, hidden_size, state_dim, partial=True, title="collecting recurrent"):
    state, _ = env.reset()
    if partial:
        state = state[:state_dim]
    adim = env.action_space.sample().shape[0]
    amin = env.action_space.low[0]
    amax = env.action_space.high[0]

    buffer = Buffer(state_dim, adim, size)
    h = th.zeros(1, 1, hidden_size)
    c = th.zeros(1, 1, hidden_size)
    hidden = (h, c)

    total_reward = 0

    for i in range(size):
        action, new_hidden = act_recurrent(policy, state, hidden, amin, amax)

        next_state, reward, terminated, truncated, info = env.step(action)
        if partial:
            next_state = next_state[:state_dim]
        done = terminated or truncated

        buffer.add(state, action, reward, done, next_state)
        total_reward += reward

        if done:
            state, _ = env.reset()
            if partial:
                state = state[:state_dim]
            h = th.zeros(1, 1, hidden_size)
            c = th.zeros(1, 1, hidden_size)
            hidden = (h, c)
        else:
            state = next_state
            hidden = new_hidden

    return buffer, total_reward / size  