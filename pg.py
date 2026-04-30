"""
HW4 — Entry point.

Import from the task files (buffer.py, vpg.py, gae.py, ppo.py) and run
end-to-end experiments. Uncomment the blocks below once each task is done.

File layout:
    buffer.py  — Task 1: Buffer, collect_data, act, rescale_actions
    vpg.py     — Task 2: reinforce_signal, train_vpg
    gae.py     — Task 3: compute_gae, reinforce_adv_signal, train_advantage_vpg
    ppo.py     — Tasks 4 & 5: ppo_surrogate_loss, ppo_total_loss, train_ppo

Tip: Tasks 1-4 are largely independent and can be split across the team.
  - Task 1 (buffer + env loop + reward-to-go) is the data-collection infra.
  - Task 2 (vanilla PG losses) can be written once the buffer interface is agreed.
  - Task 3 adds a critic and GAE (also parallelisable).
  - Task 4 is just two loss functions — fully independent of the buffer.
  - Task 5 glues everything together into PPO.

The plotting utilities are in plotting.py — you should not need to implement
them. Each training function returns (policy, returns, ...) so you can pass
the metrics directly to the plotters and the policy to record_video.
"""

from vpg import train_vpg                        
from gae import train_advantage_vpg              
from ppo import train_ppo                        
from plotting import plot_learning_curves, plot_loss_curves
from video import record_video, generate_strobe  


if __name__ == "__main__":

    # --- Task 2: vanilla PG at two learning rates ---
    policy_lo, ret_lo = train_vpg(epochs=200, learning_rate=1e-4)
    policy_hi, ret_hi = train_vpg(epochs=200, learning_rate=3e-4)
    plot_learning_curves(
        {"lr=1e-4": ret_lo, "lr=3e-4": ret_hi},
        title="Task 2: VPG with different learning rates",
    )
    record_video(policy_hi, path="videos/task2_vpg.mp4")  # optional

    # --- Task 3: rewards-to-go vs GAE ---
    policy_rtg, ret_rtg = train_vpg(epochs=200, learning_rate=3e-4)
    policy_gae, ret_gae = train_advantage_vpg(epochs=200, learning_rate=3e-4)
    plot_learning_curves(
        {"rewards-to-go": ret_rtg, "GAE": ret_gae},
        title="Task 3: rewards-to-go vs GAE",
    )
    record_video(policy_gae, path="videos/task3_gae.mp4")  # optional

    # --- Task 4: PPO surrogate clipped vs unclipped ---
    _, ret_clip, loss_clip = train_ppo(iterations=200, clip=True)
    _, ret_noclip, loss_noclip = train_ppo(iterations=200, clip=False)
    plot_learning_curves(
        {"clipped": ret_clip, "unclipped": ret_noclip},
        title="Task 4: PPO learning curves",
    )
    plot_loss_curves(
        {"clipped": loss_clip, "unclipped": loss_noclip},
        title="Task 4: PPO loss curves",
    )

    # --- Task 5: full PPO ---
    policy, ret_ppo, loss_ppo = train_ppo(iterations=500)
    plot_learning_curves({"PPO": ret_ppo}, title="Task 5: Full PPO")
    plot_loss_curves({"PPO": loss_ppo}, title="Task 5: Total loss")
    record_video(policy, path="videos/task5_ppo.mp4")            # optional
    generate_strobe(policy, path="videos/task5_ppo_strobe.png")  # optional

    pass
