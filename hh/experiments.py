"""Simulation entry point, analysis helpers and experiment protocols."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hh.config import SOLVER, SPIKE_THRESHOLD_MV
from hh.model import HodgkinHuxley
from hh.solvers import Solution, rk4
from hh.stimuli import ConstantCurrent

FloatArray = NDArray[np.float64]


def simulate(
    model: HodgkinHuxley,
    t_span: tuple[float, float],
    y0: ArrayLike | None = None,
    dt: float = SOLVER.rk4_dt,
    store_every: int = 1,
) -> Solution:
    """Integrate ``model`` with RK4, restarting at every stimulus breakpoint.

    The interval is split at the stimulus's ``breakpoints``. On each piece the
    current is frozen at its value at the piece's midpoint, so the ODE solved
    there is smooth and RK4 keeps its 4th-order accuracy. Each piece uses the
    largest step <= ``dt`` that divides it evenly.

    Args:
        model: the HH model; its stimulus may return per-trial arrays.
        t_span: (t0, t1) in ms.
        y0: initial state; defaults to the resting state (broadcast over trials
            if the stimulus is batched).
        dt: maximum step size (ms).
        store_every: forwarded to the solver.
    """
    t0, t1 = map(float, t_span)
    if y0 is None:
        y0 = model.resting_state()
    breakpoints = getattr(model.stimulus, "breakpoints", ())
    edges = [t0, *sorted(b for b in breakpoints if t0 < b < t1), t1]

    # A batched stimulus (one current per trial) needs one state column per trial.
    pieces = list(zip(edges[:-1], edges[1:]))
    batch_shape = np.broadcast_shapes(*(np.shape(model.stimulus(0.5 * (a + b))) for a, b in pieces))
    y = np.asarray(y0, dtype=float)
    if y.ndim == 1 and batch_shape:
        y = np.broadcast_to(y.reshape((-1,) + (1,) * len(batch_shape)), y.shape + batch_shape).copy()

    ts: list[FloatArray] = []
    ys: list[FloatArray] = []
    nfev = 0
    for a, b in pieces:
        current = model.stimulus(0.5 * (a + b))
        piece = replace(model, stimulus=ConstantCurrent(current))
        n_steps = max(1, math.ceil((b - a) / dt - 1e-9))
        sol = rk4(piece.rhs, (a, b), y, (b - a) / n_steps, store_every)
        skip = 0 if not ts else 1  # the first point of a piece duplicates the last of the previous one
        ts.append(sol.t[skip:])
        ys.append(sol.y[..., skip:])
        nfev += sol.nfev
        y = sol.y[..., -1]

    return Solution(t=np.concatenate(ts), y=np.concatenate(ys, axis=-1), nfev=nfev, method="RK4")


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
