"""Shared figure style and reusable plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts only save files; no display needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from hh.config import FIGURES_DIR, PLOT_RC, STATE_COLORS


def apply_style() -> None:
    """Apply the project-wide matplotlib style from hh.config."""
    plt.rcParams.update(PLOT_RC)


def save_figure(fig: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Save ``fig`` as ``<directory>/<name>.png`` and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_state_trace(
    t: ArrayLike,
    y: ArrayLike,
    I_ext: ArrayLike | None = None,
    title: str | None = None,
) -> Figure:
    """V(t) on top, gating variables m, n, h underneath, optional I_ext strip at the bottom.

    Args:
        t: times (ms), shape (n_t,).
        y: states, shape (4, n_t).
        I_ext: injected current (uA/cm^2) sampled at ``t``.
    """
    t = np.asarray(t)
    V, m, n, h = np.asarray(y)
    rows = 3 if I_ext is not None else 2
    ratios = [3, 2, 1] if I_ext is not None else [3, 2]
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(8, 5.5), height_ratios=ratios)

    axes[0].plot(t, V, color=STATE_COLORS["V"])
    axes[0].set_ylabel("V (mV)")
    for name, x in zip("mnh", (m, n, h)):
        axes[1].plot(t, x, color=STATE_COLORS[name], label=name)
    axes[1].set_ylabel("gating variable")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if I_ext is not None:
        axes[2].plot(t, np.broadcast_to(I_ext, t.shape), color="0.3")
        axes[2].set_ylabel(r"$I_{ext}$ ($\mu$A/cm$^2$)")
    axes[-1].set_xlabel("time (ms)")
    if title:
        axes[0].set_title(title)
    fig.align_ylabels(axes)
    return fig
