"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.

Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""

import numpy as np
import torch as th
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor
from gae import build_critic, compute_gae, critic_loss


# ---------------------------------------------------------------------------
# Internal helper (provided — do not modify)
# ---------------------------------------------------------------------------

def _critic_values(critic, buffer):
    """Run the critic on every stored state, returning (values, next_values)."""
    states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
    with th.no_grad():
        values = critic(states).numpy()
    next_values = np.zeros_like(values)
    next_values[:-1] = values[1:]
    return values, next_values


# ---------------------------------------------------------------------------
# Task 4 TODOs
# ---------------------------------------------------------------------------

def ppo_surrogate_loss(
    policy, states, actions, advantages, old_log_probs, eps_clip=0.2, clip=True
):
    """PPO surrogate objective (PPO paper, Equation 7).

        r_t(theta) = exp( log pi_theta(a|s) - log pi_theta_old(a|s) )

        unclipped:   L = E[ r_t * A_t ]
        clipped:     L = E[ min( r_t * A_t,
                                 clip(r_t, 1 - eps, 1 + eps) * A_t ) ]

    Returns the *negative* of the objective so that optimizer.step()
    performs gradient ascent on the expected return.
    """
    r_theta = th.exp(_log_prob(policy, states, actions) - old_log_probs)

    
    if (not clip):  #unclipped
        L = r_theta * advantages
    else:  #clipped
        L = th.min(r_theta * advantages, th.clamp(r_theta, 1 - eps_clip, 1 + eps_clip) * advantages)

    return -L.mean()


def ppo_total_loss(
    policy,
    critic,
    states,
    actions,
    advantages,
    returns,
    old_log_probs,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    clip=True,
):
    """PPO total loss (PPO paper, Equation 9).

        L_total = L_surr  +  c1 * L_VF  -  c2 * S[pi]

    where L_VF = ( V_theta(s) - R_t )^2  and  S[pi] is the policy entropy.
    Returns a scalar tensor to be minimised.
    """
    L_surr = ppo_surrogate_loss(policy, states, actions, advantages, old_log_probs, eps_clip, clip)

    mu, sigma = policy(states)

    v_theta = critic(states)
    L_VF = mse_loss(v_theta, returns)

    s_pi = Normal(mu, sigma).entropy().mean()

    L_total = L_surr + c1 * L_VF - c2 * s_pi

    return L_total


# ---------------------------------------------------------------------------
# Task 5 TODO
# ---------------------------------------------------------------------------

def train_ppo(
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
):
    """Full PPO algorithm with a single actor (N = 1).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-iteration average episodic returns
        losses  — list of per-iteration total loss values
    """
    env = gym.make("Pendulum-v1")
    state_dim  = env.reset()[0].shape[0]
    action_dim = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy       = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = th.optim.Adam(policy.parameters(), lr=learning_rate)
    cr_optimizer = th.optim.Adam(critic.parameters(), lr=learning_rate)

    returns_per_iter = []
    losses_per_iter  = []

    for k in range(iterations):
        # TODO: 1) roll out the current policy for `steps_per_iter` steps
        #          and store transitions in a Buffer.
        #       2) compute V(s) and V(s') with the critic, then GAE advantages
        #          and target returns (returns = advantages + V(s)).
        #       3) cache the log-probabilities of the sampled actions under
        #          the *old* policy (detach from the graph).
        #       4) for `sgd_epochs` epochs, iterate over minibatches of the
        #          collected data and minimise ppo_total_loss(...).
        #       5) log per-iteration episodic return and total loss for the
        #          required learning / loss curve plots.
        with th.no_grad():
            buffer, avg_rwd = collect_data(
                steps_per_iter, env, policy, title=f"ppo"
            )

        buffer.calc_reward_to_go()

        all_states_t = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
        all_actions_t = th.as_tensor(buffer.actions[: buffer.max_i], dtype=th.float32)
        with th.no_grad():
            values = critic(all_states_t).numpy()   
                # V(s_t)
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]                    # V(s_{t+1}), 0 at episode end
        all_rewards = buffer.rewards[: buffer.max_i]
        all_dones = buffer.dones[: buffer.max_i]

        advantages = compute_gae(all_rewards, values, next_values, all_dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_advantages_t = th.as_tensor(advantages, dtype=th.float32)


        old_log_probs = _log_prob(policy=policy, states=all_states_t, actions=all_actions_t).detach()
        
        all_returns_t = th.as_tensor(advantages + values, dtype=th.float32)
        iter_loss = []
        for _ in range(sgd_epochs):

            idxs = np.random.randint(0, buffer.max_i, size=minibatch_size)
            states_t    = all_states_t[idxs]
            actions_t   = all_actions_t[idxs]
            adv_t       = all_advantages_t[idxs]
            returns_t   = all_returns_t[idxs]
            old_lp_t    = old_log_probs[idxs]

            cr_optimizer.zero_grad()
            optimizer.zero_grad()


            ppo_loss = ppo_total_loss(
                policy=policy, 
                critic=critic, 
                states=states_t, 
                actions=actions_t, 
                advantages=adv_t,
                returns=returns_t,
                old_log_probs=old_lp_t,
                eps_clip=eps_clip,
                c1=c1,
                c2=c2,
                clip=clip,
            )
            iter_loss.append(ppo_loss.item())
            
            ppo_loss.backward()
            optimizer.step()
            cr_optimizer.step()

        returns_per_iter.append(avg_rwd * episode_len)
        losses_per_iter.append(np.mean(iter_loss))
    
    return policy, returns_per_iter, losses_per_iter
        





    # TODO: return policy, list_of_returns, list_of_losses


if __name__ == "__main__":
    from plotting import plot_learning_curves, plot_loss_curves
    from video import record_video, generate_strobe

    # --- Task 4: clipped vs unclipped ---
    # _, ret_clip,   loss_clip   = train_ppo(iterations=50, clip=True)
    # _, ret_noclip, loss_noclip = train_ppo(iterations=50, clip=False)
    # plot_learning_curves(
    #     {"clipped": ret_clip, "unclipped": ret_noclip},
    #     title="Task 4: PPO clipped vs unclipped",
    # )
    # plot_loss_curves(
    #     {"clipped": loss_clip, "unclipped": loss_noclip},
    #     title="Task 4: PPO loss curves",
    # )

    # --- Task 5: full PPO ---
    policy, ret_ppo, loss_ppo = train_ppo(iterations=500)
    plot_learning_curves({"PPO": ret_ppo}, title="Task 5: Full PPO")
    plot_loss_curves({"PPO": loss_ppo}, title="Task 5: Total loss")
    record_video(policy, path="videos/task5_ppo.mp4")            # optional
    generate_strobe(policy, path="videos/task5_ppo_strobe.png")  # optional
