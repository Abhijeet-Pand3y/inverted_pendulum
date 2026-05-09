"""Plotting utilities for HW4.

These functions consume per-epoch (or per-iteration) metrics produced by
the training routines in `pg.py` and generate the plots required by the
handout. You do NOT need to implement anything in this file — just import
and call these helpers from your experiments.

Pass `smooth` in (0, 1) to overlay an EMA-smoothed trend line on top of
the raw curve. 0.9 is a reasonable default for noisy RL learning curves;
0.0 disables smoothing.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def _ema(values, decay):
    """Exponential moving average. `decay=0` returns the raw series."""
    values = np.asarray(values, dtype=np.float64)
    if decay <= 0 or len(values) == 0:
        return values
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = decay * out[i - 1] + (1.0 - decay) * values[i]
    return out


def _sanitize_filename(title):
    """Convert title to a safe filename."""
    return title.lower().replace(" ", "_").replace(":", "").replace("/", "_")


# ---------------------------------------------------------------------------
# Original functions — display plots (unchanged)
# ---------------------------------------------------------------------------

def plot_learning_curve(
    returns, title="Learning curve", ylabel="Episodic return",
    save_path=None, smooth=0.0,
):
    """Plot a single learning curve (one list/array of per-epoch values)."""
    plot_learning_curves(
        {"run": returns}, title=title, ylabel=ylabel,
        save_path=save_path, smooth=smooth,
    )


def plot_learning_curves(
    curves, title="Learning curves", ylabel="Episodic return",
    save_path=None, smooth=0.0,
):
    """Plot multiple labelled curves on the same axes."""
    plt.figure()
    for label, values in curves.items():
        values = np.asarray(values, dtype=np.float64)
        if smooth > 0:
            line, = plt.plot(values, alpha=0.25)
            plt.plot(_ema(values, smooth), label=label, color=line.get_color())
        else:
            plt.plot(values, label=label)
    plt.xlabel("Training epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()


def plot_loss_curves(losses, title="Loss curves", save_path=None, smooth=0.0):
    """Plot one or more loss curves. `losses` is a dict label -> iterable."""
    plot_learning_curves(
        losses, title=title, ylabel="Loss",
        save_path=save_path, smooth=smooth,
    )


# ---------------------------------------------------------------------------
# Save variants — auto-save to ./result_images, no display
# ---------------------------------------------------------------------------

SAVE_DIR = "./result_images"


def save_learning_curve(
    returns, title="Learning curve", ylabel="Episodic return",
    smooth=0.0,
):
    """Save a single learning curve to result_images/."""
    save_learning_curves(
        {"run": returns}, title=title, ylabel=ylabel, smooth=smooth,
    )


def save_learning_curves(
    curves, title="Learning curves", ylabel="Episodic return",
    smooth=0.0,
):
    """Save multiple labelled curves to result_images/."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = os.path.join(SAVE_DIR, _sanitize_filename(title) + ".png")

    plt.figure()
    for label, values in curves.items():
        values = np.asarray(values, dtype=np.float64)
        if smooth > 0:
            line, = plt.plot(values, alpha=0.25)
            plt.plot(_ema(values, smooth), label=label, color=line.get_color())
        else:
            plt.plot(values, label=label)
    plt.xlabel("Training epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {filename}")


def save_loss_curves(losses, title="Loss curves", smooth=0.0):
    """Save one or more loss curves to result_images/."""
    save_learning_curves(
        losses, title=title, ylabel="Loss", smooth=smooth,
    )