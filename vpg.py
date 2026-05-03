"""
HW4 — Task 2: Vanilla policy gradient (REINFORCE).

Depends on: buffer.py (Task 1 must be complete).
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

from Modules import NormalModule
from buffer import Buffer, collect_data, act, rescale_actions


# ---------------------------------------------------------------------------
# Shared helpers (provided — do not modify)
# ---------------------------------------------------------------------------

def _log_prob(policy, states, actions):
    """Compute sum of log-probabilities under the current policy."""
    mu, sigma = policy(states)
    return Normal(mu, sigma).log_prob(actions).sum(dim=-1, keepdim=True)


def build_actor(state_dim, action_dim, hidden_size):
    """Two-layer feed-forward actor ending in NormalModule (provided).

    Architecture:
        Linear(state_dim, hidden_size) -> ReLU
        -> Linear(hidden_size, hidden_size) -> ReLU
        -> NormalModule(hidden_size, action_dim)
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        NormalModule(hidden_size, action_dim),
    )


# ---------------------------------------------------------------------------
# Task 2 TODOs
# ---------------------------------------------------------------------------

def reinforce_signal(policy, states, actions, rewards_to_go, avg_rwd, use_avg=False):
    """Vanilla policy-gradient loss weighted by reward-to-go."""
    # TODO: compute  -E[ (R_to_go - baseline?) * log pi(a | s) ]
    log_probs = _log_prob(policy, states, actions)

    if use_avg:
        b = avg_rwd
    else:
        b = 0

    L = log_probs * (rewards_to_go - b)

    return -L.mean()


def reinforce_rwd_signal(policy, states, actions, rewards):
    """REINFORCE loss using one-step rewards instead of reward-to-go."""
    # TODO: compute  -E[ r_t * log pi(a | s) ].
    log_probs = _log_prob(policy, states, actions)
    return -(log_probs * rewards).mean()


def train_vpg(
    epochs=3,
    episodes=10,
    updates=10,
    learning_rate=1e-4,
    hidden_size=32,
    layers=2,
    batch_size=512,
    use_avg=False,
    use_rwds=False,
    gamma=0.975,
):
    """Train the vanilla policy-gradient agent (Task 2).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-epoch average episodic returns
    """
    env = gym.make("Pendulum-v1")
    state_dim  = env.reset()[0].shape[0]
    action_dim = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy    = build_actor(state_dim, action_dim, hidden_size)
    optimizer = th.optim.Adam(params=policy.parameters(), lr=learning_rate)

    returns_per_epoch = []
    for x in range(epochs):
        # TODO: 1) roll out to fill a buffer (use collect_data under th.no_grad)
        #       2) buffer.calc_reward_to_go()
        with th.no_grad():
            buffer, avg_rwd = collect_data(episodes * episode_len, env, policy)
        
        buffer.calc_reward_to_go(gamma=gamma)

        for i in range(updates):
            # TODO: You need to sample from the buffer here
            s, a, r, ns, d, rtg, nrtg = buffer.sample(batch_size)
            
            # TODO: After sampling you need to convert numpy arrays to tensors, Example: "s_t = th.as_tensor(s, dtype=th.float32)"
            
            s_t = th.as_tensor(s, dtype=th.float32)
            a_t = th.as_tensor(a, dtype=th.float32)
            r_t = th.as_tensor(r, dtype=th.float32)
            rtg_t = th.as_tensor(rtg, dtype=th.float32)

            amin = env.action_space.low[0]
            amax = env.action_space.high[0]
            a_t = 2 * (a_t - amin) / (amax - amin) - 1

            optimizer.zero_grad()

            # TODO: compute loss here
            if use_rwds:
                loss = reinforce_rwd_signal(policy, s_t, a_t, r_t)
            else:
                loss = reinforce_signal(policy, s_t, a_t, rtg_t, avg_rwd, use_avg)

            loss.backward()
            optimizer.step()

        # TODO: record the epoch's avg episodic return for the learning curve.
        ep_returns = []
        ep_return = 0.0
        discount = 1.0

        for i in range(buffer.max_i):
            ep_return += discount * buffer.rewards[i, 0]
            discount *= gamma

            if buffer.dones[i, 0]:
                ep_returns.append(ep_return)
                ep_return = 0.0
                discount = 1.0

        if len(ep_returns) > 0:
            returns_per_epoch.append(np.mean(ep_returns))
        else:
            returns_per_epoch.append(ep_return)

    # TODO: return (policy, list_of_per_epoch_returns).
    env.close()
    return policy, returns_per_epoch


if __name__ == "__main__":
    from plotting import plot_learning_curves
    from video import record_video

    # Example: compare two learning rates
    policy_lo, ret_lo = train_vpg(epochs=200, learning_rate=1e-4)
    policy_hi, ret_hi = train_vpg(epochs=200, learning_rate=3e-4)
    plot_learning_curves(
        {"lr=1e-4": ret_lo, "lr=3e-4": ret_hi},
        title="Task 2: VPG with different learning rates",
    )
    record_video(policy_hi, path="videos/task2_vpg.mp4")  # optional