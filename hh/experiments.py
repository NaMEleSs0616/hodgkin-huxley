"""Analysis helpers and experiment protocols built on the model and solvers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hh.config import SPIKE_THRESHOLD_MV

FloatArray = NDArray[np.float64]


def spike_times(t: ArrayLike, V: ArrayLike, threshold: float = SPIKE_THRESHOLD_MV) -> FloatArray:
    """Times of upward crossings of ``threshold``, located by linear interpolation.

    Args:
        t: times, shape (n_t,).
        V: membrane potential for a single trial, shape (n_t,).
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    above = V >= threshold
    idx = np.flatnonzero(~above[:-1] & above[1:])  # V[i] < thr <= V[i+1]
    frac = (threshold - V[idx]) / (V[idx + 1] - V[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])
