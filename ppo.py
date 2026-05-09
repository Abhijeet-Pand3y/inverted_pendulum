"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.

Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""

import numpy as np
import torch as th
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym
import time
from lstm import RecurrentActor

from buffer import Buffer, collect_data, act, rescale_actions, collect_data_parallel, collect_data_recurrent
from vpg import _log_prob, build_actor
from gae import build_critic, compute_gae, critic_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env():
    return gym.make("Pendulum-v1")


def _critic_values(critics, buffer):
    """Run ensemble of critics on stored states. Returns (values, next_values).

    Works for both sequential and parallel collection because it uses
    buffer.next_states directly instead of relying on storage order.
    """
    states      = th.as_tensor(buffer.states[: buffer.max_i],      dtype=th.float32)
    next_states = th.as_tensor(buffer.next_states[: buffer.max_i], dtype=th.float32)
    with th.no_grad():
        values      = th.stack([c(states)      for c in critics]).mean(dim=0).numpy()
        next_values = th.stack([c(next_states) for c in critics]).mean(dim=0).numpy()
    return values, next_values


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def ppo_surrogate_loss(policy, states, actions, advantages, old_log_probs,
                      eps_clip=0.2, clip=True):
    """PPO surrogate objective (PPO paper, Equation 7).

        r_t(theta) = exp( log pi_theta(a|s) - log pi_theta_old(a|s) )

        unclipped: L = E[ r_t * A_t ]
        clipped:   L = E[ min( r_t * A_t,
                               clip(r_t, 1-eps, 1+eps) * A_t ) ]
    """
    r_theta = th.exp(_log_prob(policy, states, actions) - old_log_probs)

    if not clip:
        L = r_theta * advantages
    else:
        L = th.min(
            r_theta * advantages,
            th.clamp(r_theta, 1 - eps_clip, 1 + eps_clip) * advantages
        )

    return -L.mean()


def ppo_total_loss(policy, critic, states, actions, advantages, returns,
                  old_log_probs, eps_clip=0.2, c1=0.5, c2=0.01, clip=True):
    """PPO total loss (PPO paper, Equation 9).

        L_total = L_surr + c1 * L_VF - c2 * S[pi]
    """
    L_surr = ppo_surrogate_loss(policy, states, actions, advantages,
                                old_log_probs, eps_clip, clip)
    mu, sigma = policy(states)
    L_VF = mse_loss(critic(states), returns)
    s_pi = Normal(mu, sigma).entropy().mean()
    return L_surr + c1 * L_VF - c2 * s_pi


def ppo_ensemble_loss(policy, states, actions, advantages, old_log_probs,
                     c2=0.01, eps_clip=0.2, clip=True):
    """PPO loss for ensemble setting — critic loss handled separately.

    L_total = L_surr - c2 * S[pi]
    """
    L_surr = ppo_surrogate_loss(policy, states, actions, advantages,
                                old_log_probs, eps_clip, clip)
    mu, sigma = policy(states)
    s_pi = Normal(mu, sigma).entropy().mean()
    return L_surr - c2 * s_pi

def _log_prob_recurrent_seq(policy, states_seq, actions, hidden_size):
    # states_seq: (batch, seq_len, state_dim)
    # actions: (batch, action_dim) — only last step
    batch_size = states_seq.shape[0]
    h = th.zeros(1, batch_size, hidden_size)
    c = th.zeros(1, batch_size, hidden_size)
    mu, sigma, _ = policy(states_seq, (h, c))  # mu shape: (batch, action_dim)
    return Normal(mu, sigma).log_prob(actions).sum(dim=-1, keepdim=True)

def ppo_ensemble_loss_recurrent(policy, states, actions, advantages, 
                                 old_log_probs, hidden_size, c2=0.01, eps_clip=0.2):
    r_theta = th.exp(_log_prob_recurrent_seq(policy, states, actions, hidden_size) - old_log_probs)
    L = th.min(r_theta * advantages,
               th.clamp(r_theta, 1 - eps_clip, 1 + eps_clip) * advantages)
    L_surr = -L.mean()
    batch_size = states.shape[0]
    h = th.zeros(1, batch_size, hidden_size)
    c = th.zeros(1, batch_size, hidden_size)
    mu, sigma, _ = policy(states, (h, c))  # states already (batch, seq_len, state_dim)
    s_pi = Normal(mu, sigma).entropy().mean()
    return L_surr - c2 * s_pi



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ppo(
    # Core hyperparameters
    iterations=200,
    steps_per_iter=2048,
    sgd_epochs=10,
    minibatch_size=64,
    learning_rate=3e-4,
    hidden_size=64,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    clip=True,
    # Extension flags
    use_parallel=False,    # parallel data collection
    num_envs=8,            # only used if use_parallel=True
    use_ensemble=False,    # ensemble of B critics
    B=3,                   # only used if use_ensemble=True
):
    """Full PPO algorithm with optional extensions.

    Extensions toggled via flags:
      - use_parallel: parallel data collection across `num_envs` environments
      - use_ensemble: ensemble of B critics (averaged for inference,
                      trained independently)

    Returns:
        policy  — trained actor network
        returns — list of per-iteration episodic returns
        losses  — list of per-iteration total loss values
    """

    # ---- Environment setup ----
    if use_parallel:
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
        states_dim  = envs.reset()[0].shape[1]
        actions_dim = envs.single_action_space.sample().shape[0]
        episode_len = envs.get_attr("spec")[0].max_episode_steps
    else:
        env = make_env()
        states_dim  = env.reset()[0].shape[0]
        actions_dim = env.action_space.sample().shape[0]
        episode_len = env.spec.max_episode_steps

    # ---- Policy + optimizer ----
    policy    = build_actor(states_dim, actions_dim, hidden_size)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)

    # ---- Critic(s) + optimizer(s) ----
    if use_ensemble:
        critics       = [build_critic(states_dim, hidden_size) for _ in range(B)]
        cr_optimizers = [th.optim.Adam(c.parameters(), lr=learning_rate)
                         for c in critics]
    else:
        critic       = build_critic(states_dim, hidden_size)
        cr_optimizer = th.optim.Adam(critic.parameters(), lr=learning_rate)

    # ---- Logging ----
    returns_per_iter = []
    losses_per_iter  = []
    time_taken       = []

    # ---- Training loop ----
    for k in range(iterations):

        # 1) Collect data
        with th.no_grad():
            start = time.time()
            if use_parallel:
                buffer, avg_rwd = collect_data_parallel(
                    steps_per_iter // num_envs, envs, num_envs, policy
                )
            else:
                buffer, avg_rwd = collect_data(steps_per_iter, env, policy)
            time_taken.append(time.time() - start)

        # 2) Compute V(s), V(s'), GAE advantages, returns
        buffer.calc_reward_to_go(gamma)

        if use_ensemble:
            values, next_values = _critic_values(critics, buffer)
        else:
            values, next_values = _critic_values([critic], buffer)

        all_rewards = buffer.rewards[: buffer.max_i]
        all_dones   = buffer.dones[: buffer.max_i]

        advantages = compute_gae(all_rewards, values, next_values,
                                all_dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3) Cache old log probs and convert tensors
        all_states_t  = th.as_tensor(buffer.states[: buffer.max_i],
                                    dtype=th.float32)
        all_actions_t = th.as_tensor(buffer.actions[: buffer.max_i],
                                    dtype=th.float32)
        all_adv_t     = th.as_tensor(advantages,           dtype=th.float32)
        all_returns_t = th.as_tensor(advantages + values,  dtype=th.float32)
        old_log_probs = _log_prob(policy, all_states_t, all_actions_t).detach()

        # 4) Inner SGD loop
        iter_loss = []
        for _ in range(sgd_epochs):
            idxs       = np.random.randint(0, buffer.max_i, size=minibatch_size)
            states_t   = all_states_t[idxs]
            actions_t  = all_actions_t[idxs]
            adv_t      = all_adv_t[idxs]
            returns_t  = all_returns_t[idxs]
            old_lp_t   = old_log_probs[idxs]

            # Update critic(s)
            if use_ensemble:
                for b in range(B):
                    cr_optimizers[b].zero_grad()
                    loss_b = mse_loss(critics[b](states_t), returns_t)
                    loss_b.backward()
                    th.nn.utils.clip_grad_norm_(critics[b].parameters(),
                                               max_norm=0.5)
                    cr_optimizers[b].step()

            # Update policy (and single critic, if not ensemble)
            optimizer.zero_grad()
            if use_ensemble:
                ppo_loss = ppo_ensemble_loss(
                    policy, states_t, actions_t, adv_t, old_lp_t,
                    c2=c2, eps_clip=eps_clip, clip=clip,
                )
            else:
                cr_optimizer.zero_grad()
                ppo_loss = ppo_total_loss(
                    policy, critic, states_t, actions_t, adv_t, returns_t,
                    old_lp_t, eps_clip=eps_clip, c1=c1, c2=c2, clip=clip,
                )

            ppo_loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            if not use_ensemble:
                th.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.3)
                cr_optimizer.step()

            iter_loss.append(ppo_loss.item())

        # 5) Logging
        returns_per_iter.append(avg_rwd * episode_len)
        losses_per_iter.append(np.mean(iter_loss))
        # print(f"iter {k+1}/{iterations}  return={returns_per_iter[-1]:.2f}  "
        #       f"loss={losses_per_iter[-1]:.4f}  time={time_taken[-1]:.2f}s")

    print(f"\nTotal collection time: {sum(time_taken):.2f}s")
    print(f"Average per-iter time: {sum(time_taken)/iterations:.3f}s")
    return policy, returns_per_iter, losses_per_iter


def train_ppo_recurrent(
    # Core hyperparameters
    iterations=200,
    steps_per_iter=2048,
    sgd_epochs=10,
    minibatch_size=64,
    learning_rate=3e-4,
    hidden_size=64,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    use_ensemble=False,
    B=3,
    clip=True,
    # Recurrent specific
    state_dim=3,       # partial observability — no theta_dot
    action_dim=1,
    seq_len=16,
    partial=True,
):
    env = make_env()
    actions_dim = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy = RecurrentActor(state_dim=state_dim, action_dim=1, hidden_size=hidden_size)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)

    if use_ensemble:
        critics       = [build_critic(state_dim, hidden_size) for _ in range(B)]
        cr_optimizers = [th.optim.Adam(c.parameters(), lr=learning_rate)
                         for c in critics]
    else:
        critic       = build_critic(state_dim, hidden_size)
        cr_optimizer = th.optim.Adam(critic.parameters(), lr=learning_rate)

    returns_per_iter = []
    losses_per_iter  = []
    time_taken = []

    for k in range(iterations):

        with th.no_grad():
            start = time.time()
            buffer, avg_rwd  = collect_data_recurrent(
                steps_per_iter, env, policy, hidden_size, state_dim, True
            )
            time_taken.append(time.time()-start)

        buffer.calc_reward_to_go(gamma)

        if use_ensemble:
                values, next_values = _critic_values(critics, buffer)
        else:
                values, next_values = _critic_values([critic], buffer)

        all_rewards = buffer.rewards[: buffer.max_i]
        all_dones   = buffer.dones[: buffer.max_i]

        advantages = compute_gae(all_rewards, values, next_values,
                                all_dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3) Cache old log probs and convert tensors
        all_states_t  = th.as_tensor(buffer.states[: buffer.max_i],
                                    dtype=th.float32)
        all_actions_t = th.as_tensor(buffer.actions[: buffer.max_i],
                                    dtype=th.float32)
        all_adv_t     = th.as_tensor(advantages,           dtype=th.float32)
        all_returns_t = th.as_tensor(advantages + values,  dtype=th.float32)
        

        iter_loss = []

        for _ in range(sgd_epochs):
            
            
            # sample random start points, ensure sequence fits in buffer
            starts = np.random.randint(0, buffer.max_i - seq_len, size=minibatch_size)
            # build index array: shape (minibatch_size, seq_len)
            idxs = np.array([np.arange(s, s + seq_len) for s in starts])

            states_t  = all_states_t[idxs]  # (batch, seq_len, state_dim)

            actions_t = all_actions_t[idxs[:, -1]]   # (batch, action_dim)
            adv_t     = all_adv_t[idxs[:, -1]]
            returns_t = all_returns_t[idxs[:, -1]]
            with th.no_grad():
                old_lp_t = _log_prob_recurrent_seq(policy, states_t, actions_t, hidden_size).detach()

            # Update critic(s)
            if use_ensemble:
                for b in range(B):
                    cr_optimizers[b].zero_grad()
                    loss_b = mse_loss(critics[b](states_t[:, -1, :]), returns_t)
                    loss_b.backward()
                    th.nn.utils.clip_grad_norm_(critics[b].parameters(),
                                               max_norm=0.5)
                    cr_optimizers[b].step()

            # Update policy (and single critic, if not ensemble)
            if not use_ensemble:
                cr_optimizer.zero_grad()
                loss_b = loss_b = mse_loss(critic(states_t[:, -1, :]), returns_t)
                loss_b.backward()
                th.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                cr_optimizer.step()

            optimizer.zero_grad()
            ppo_loss = ppo_ensemble_loss_recurrent(
                    policy, states_t, actions_t, adv_t, old_lp_t,
                    hidden_size=hidden_size, c2=c2, eps_clip=eps_clip,
                )

            ppo_loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            

            iter_loss.append(ppo_loss.item())

        # 5) Logging
        returns_per_iter.append(avg_rwd * episode_len)
        losses_per_iter.append(np.mean(iter_loss))
        print(f"iter {k+1}/{iterations}  return={returns_per_iter[-1]:.2f}  "
              f"loss={losses_per_iter[-1]:.4f}  time={time_taken[-1]:.2f}s")

    print(f"\nTotal collection time: {sum(time_taken):.2f}s")
    print(f"Average per-iter time: {sum(time_taken)/iterations:.3f}s")
    return policy, returns_per_iter, losses_per_iter



if __name__ == "__main__":
    # from plotting import plot_learning_curves, plot_loss_curves
    from plotting import save_learning_curves as plot_learning_curves, save_loss_curves as plot_loss_curves
    from video import record_video, generate_strobe

    # ---- Task 4: clipped vs unclipped ----
    _, ret_clip,   loss_clip   = train_ppo(iterations=50, clip=True)
    _, ret_noclip, loss_noclip = train_ppo(iterations=50, clip=False)
    plot_learning_curves(
        {"clipped": ret_clip, "unclipped": ret_noclip},
        title="Task 4: PPO Clipped vs Unclipped",
    )
    plot_loss_curves(
        {"clipped": loss_clip, "unclipped": loss_noclip},
        title="Task 4: Loss Clipped vs Unclipped",
    )

    # ---- Task 5: full PPO baseline ----
    policy_base, ret_base, loss_base = train_ppo(iterations=500)
    plot_learning_curves({"PPO baseline": ret_base}, title="Task 5: Full PPO Baseline")
    plot_loss_curves({"PPO baseline": loss_base},    title="Task 5: Full PPO Loss")
    record_video(policy_base, path="videos/task5_baseline.mp4")

    # ---- Extension 1: state-dependent sigma vs fixed ----
    # run with fixed sigma (current NormalModule default)
    _, ret_fixed, loss_fixed = train_ppo(
        iterations=500, learning_rate=1e-5, hidden_size=128,
        minibatch_size=256, steps_per_iter=4096,
    )
    # note: state-dependent sigma result already captured from earlier runs
    # plot separately — use saved image from earlier run in report
    plot_learning_curves(
        {"fixed sigma": ret_fixed},
        title="Extension 1: State Dependent Sigma",
    )

    # ---- Extension 2: ensemble B=1 vs B=3 ----
    _, ret_b1, loss_b1 = train_ppo(
        iterations=1000, use_ensemble=False,
        hidden_size=128, minibatch_size=256,
        steps_per_iter=4096, learning_rate=1e-4,
    )
    _, ret_b3, loss_b3 = train_ppo(
        iterations=1000, use_ensemble=True, B=3,
        hidden_size=128, minibatch_size=256,
        steps_per_iter=4096, learning_rate=1e-4,
    )
    plot_learning_curves(
        {"single critic (B=1)": ret_b1, "ensemble (B=3)": ret_b3},
        title="Extension 2: Ensemble Critics B=1 vs B=3",
    )
    plot_loss_curves(
        {"single critic (B=1)": loss_b1, "ensemble (B=3)": loss_b3},
        title="Extension 2: Ensemble Loss",
    )

    # ---- Extension 3: parallel collection (ensemble + parallel) ----
    print("No Parallel")
    policy_non_par, ret_non_par, loss_non_par = train_ppo(
        iterations=800, use_parallel=False, num_envs=8,
        use_ensemble=True, B=3,
        hidden_size=128, minibatch_size=256,
        steps_per_iter=4096, learning_rate=1e-4, sgd_epochs=10,
    )


    print("Parallel Time:")
    policy_par, ret_par, loss_par = train_ppo(
        iterations=800, use_parallel=True, num_envs=8,
        use_ensemble=True, B=3,
        hidden_size=128, minibatch_size=256,
        steps_per_iter=4096, learning_rate=1e-4, sgd_epochs=10,
    )
    
    plot_learning_curves(
        {"parallel + ensemble": ret_par},
        title="Extension 3: Parallel Collection",
    )
    plot_loss_curves(
        {"parallel + ensemble": loss_par},
        title="Extension 3: Parallel Loss",
    )
    record_video(policy_par, path="videos/ext3_parallel.mp4")

    # ---- Extension 4: LSTM partial state ----
    _, ret_lstm_partial, loss_lstm_partial = train_ppo_recurrent(
        iterations=1000, steps_per_iter=4096, hidden_size=128,
        learning_rate=3e-4, seq_len=32, sgd_epochs=5,
        minibatch_size=32, state_dim=2, partial=True,
    )

    # ---- Extension 4: LSTM full state ----
    _, ret_lstm_full, loss_lstm_full = train_ppo_recurrent(
        iterations=500, steps_per_iter=4096, hidden_size=128,
        learning_rate=3e-4, seq_len=32, sgd_epochs=5,
        minibatch_size=32, state_dim=3, partial=False,
    )

    # ---- Extension 4: overlay all three LSTM comparisons ----
    plot_learning_curves(
        {
            "feedforward full state":  ret_b1,        # reuse B=1 result as FF baseline
            "LSTM full state":         ret_lstm_full,
            "LSTM partial state":      ret_lstm_partial,
        },
        title="Extension 4: Recurrent Policy Comparison",
    )
    plot_loss_curves(
        {
            "LSTM full state":    loss_lstm_full,
            "LSTM partial state": loss_lstm_partial,
        },
        title="Extension 4: Recurrent Loss",
    )