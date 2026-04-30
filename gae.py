"""
HW4 — Task 3: Critic network and Generalized Advantage Estimation (GAE).

Depends on: buffer.py (Task 1) and vpg.py (Task 2).
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor


# ---------------------------------------------------------------------------
# Shared helper (provided — do not modify)
# ---------------------------------------------------------------------------

def build_critic(state_dim, hidden_size):
    """Two-layer feed-forward critic that outputs a scalar V(s) (provided).

    Architecture:
        Linear(state_dim, hidden_size) -> ReLU
        -> Linear(hidden_size, hidden_size) -> ReLU
        -> Linear(hidden_size, 1)
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )


# ---------------------------------------------------------------------------
# Task 3 TODOs
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, next_values, dones, gamma=0.975, lam=0.95):
    """Generalized Advantage Estimation (PPO paper, Equations 11-12). https://arxiv.org/pdf/1707.06347

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    """
    pass


def reinforce_adv_signal(policy, states, actions, advantages):
    """Policy-gradient loss weighted by arbitrary advantages (e.g. GAE)."""
    # TODO: compute  -E[ A_t * log pi(a | s) ].
    pass


def train_advantage_vpg(
    epochs=3,
    episodes=10,
    updates=10,
    critic_updates=80,
    learning_rate=1e-4,
    critic_lr=3e-4,
    hidden_size=32,
    layers=2,
    batch_size=512,
    gamma=0.975,
    lam=0.95,
):
    """Policy gradient with a learned V(s) baseline and GAE (Task 3).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-epoch average episodic returns
    """
    env = gym.make("Pendulum-v1")
    state_dim   = env.reset()[0].shape[0]
    action_dim  = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy       = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = th.optim.Adam(params=policy.parameters(), lr=learning_rate)
    cr_optimizer = th.optim.Adam(params=critic.parameters(), lr=critic_lr)

    returns_per_epoch = []
    for x in range(epochs):

        # --- collect experience ---
        with th.no_grad():
            buffer, avg_rwd = collect_data(
                episodes * episode_len, env, policy, title=f"gae {x + 1}/{epochs}"
            )

        # TODO: fill buffer.ret_to_go using buffer.calc_reward_to_go(gamma).

        # --- train the critic ---
        # Regress V(s) toward the reward-to-go targets for critic_updates steps.
        for _ in range(critic_updates):
            states, _, _, _, _, rtg, _ = buffer.sample(batch_size)
            states_t = th.as_tensor(states, dtype=th.float32)
            rtg_t    = th.as_tensor(rtg,    dtype=th.float32)
            cr_optimizer.zero_grad()
            # TODO: compute mse_loss between critic(states_t) and rtg_t,
            #       then call .backward() and cr_optimizer.step().

        # --- compute GAE advantages ---
        # Run the critic (no gradients) on every stored state.
        all_states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
        with th.no_grad():
            values = critic(all_states).numpy()          # V(s_t)
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]                    # V(s_{t+1}), 0 at episode end

        # TODO: call compute_gae(...) to get an (N, 1) array of advantages.
        advantages = None  # TODO

        # Normalise for training stability (provided).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- train the actor ---
        for _ in range(updates):
            idxs      = np.random.randint(0, buffer.max_i, size=batch_size)
            states_t  = th.as_tensor(buffer.states[idxs],  dtype=th.float32)
            actions_t = th.as_tensor(buffer.actions[idxs], dtype=th.float32)
            adv_t     = th.as_tensor(advantages[idxs],     dtype=th.float32)
            optimizer.zero_grad()
            # TODO: call reinforce_adv_signal(...) to get the loss,
            #       then call .backward() and optimizer.step().

        ep_return = avg_rwd * episode_len
        returns_per_epoch.append(ep_return)
        print(f"gae epoch {x + 1}/{epochs}: return={ep_return:.2f}")

    return policy, returns_per_epoch


if __name__ == "__main__":
    from plotting import plot_learning_curves
    from video import record_video
    from vpg import train_vpg

    # Example: compare rewards-to-go vs GAE
    policy_rtg, ret_rtg = train_vpg(epochs=200, learning_rate=3e-4)
    policy_gae, ret_gae = train_advantage_vpg(epochs=200, learning_rate=3e-4)
    plot_learning_curves(
        {"rewards-to-go": ret_rtg, "GAE": ret_gae},
        title="Task 3: rewards-to-go vs GAE",
    )
    record_video(policy_gae, path="videos/task3_gae.mp4")  # optional