"""Video utilities for HW4.

These helpers let you see what your pendulum agent is actually doing. You
do NOT need to submit any videos or strobe images — they are provided so
you can sanity-check a partially or fully trained policy.

Usage:

    from video import record_video, generate_strobe

    record_video(policy, path="videos/trained.mp4")
    generate_strobe(policy, path="strobe.png", n_frames=10)

Both helpers take any PyTorch policy that returns `(mu, sigma)` from its
`forward()` — i.e. any module ending in a NormalModule, which is what
every training routine in `pg.py` produces.

Pass `deterministic=True` (default) to render the policy mean with no
exploration noise, which usually looks cleaner than the stochastic rollout.
"""

import os
import numpy as np
import gymnasium as gym
import torch as th
from torch.distributions import Normal
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def _action_from_policy(policy, state, action_space, deterministic):
    with th.no_grad():
        mu, sigma = policy(th.as_tensor(state, dtype=th.float32))
        a = mu.numpy() if deterministic else Normal(mu, sigma).sample().numpy()
    amin, amax = action_space.low, action_space.high
    return np.clip(amin + (a + 1.0) * 0.5 * (amax - amin), amin, amax)


def _rollout_frames(policy, env_id, max_steps, deterministic, seed):
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    s, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        frames.append(env.render())
        action = _action_from_policy(policy, s, env.action_space, deterministic)
        s, _, d, t, _ = env.step(action)
        if d or t:
            break
    env.close()
    return frames


def record_video(
    policy,
    path="pendulum.mp4",
    max_steps=200,
    fps=30,
    deterministic=True,
    env_id="Pendulum-v1",
    seed=None,
):
    """Render one episode of `policy` and save it to `path` as an MP4."""
    frames = _rollout_frames(policy, env_id, max_steps, deterministic, seed)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, ffmpeg_params=["-crf", "18", "-preset", "fast"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"video saved to {path}  ({len(frames)} frames)")


def generate_strobe(
    policy,
    path="strobe.png",
    n_frames=10,
    max_steps=200,
    deterministic=True,
    env_id="Pendulum-v1",
    seed=None,
):
    """Save a horizontal strip of `n_frames` evenly-spaced frames from one episode."""
    frames = _rollout_frames(policy, env_id, max_steps, deterministic, seed)
    if not frames:
        raise RuntimeError("no frames were captured")

    idxs = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    selected = [frames[i] for i in idxs]

    fig, axes = plt.subplots(1, n_frames, figsize=(2 * n_frames, 2.5))
    if n_frames == 1:
        axes = [axes]
    for ax, frame, t in zip(axes, selected, idxs):
        ax.imshow(frame)
        ax.set_title(f"t={int(t)}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"strobe saved to {path}")
