"""Simulation entry point, analysis helpers and experiment protocols."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from hh.config import SOLVER, SPIKE_THRESHOLD_MV
from hh.model import HodgkinHuxley
from hh.solvers import Solution, dopri45, rk4
from hh.stimuli import ConstantCurrent

FloatArray = NDArray[np.float64]


SCIPY_METHODS = ("RK45", "DOP853", "LSODA", "Radau", "BDF")
METHODS = ("rk4", "dopri45") + SCIPY_METHODS


def _solve_piece(
    f, t_span: tuple[float, float], y: FloatArray, method: str, dt: float, rtol: float,
    atol: float, t_eval: FloatArray | None, store_every: int, solver_kw: dict,
) -> Solution:
    """Integrate one smooth piece with the requested method."""
    a, b = t_span
    if method == "rk4":
        n_steps = max(1, math.ceil((b - a) / dt - 1e-9))
        return rk4(f, (a, b), y, (b - a) / n_steps, store_every)
    if method == "dopri45":
        return dopri45(f, (a, b), y, rtol=rtol, atol=atol, t_eval=t_eval, **solver_kw)
    if method in SCIPY_METHODS:
        shape = y.shape
        calls = 0

        def flat_rhs(t: float, yf: FloatArray) -> FloatArray:
            # Count every call ourselves: scipy's nfev omits the calls that implicit
            # methods make to build a finite-difference Jacobian.
            nonlocal calls
            calls += 1
            return f(t, yf.reshape(shape)).ravel()

        sol = solve_ivp(flat_rhs, (a, b), y.ravel(), method=method, rtol=rtol, atol=atol,
                        t_eval=t_eval, **solver_kw)
        if sol.status != 0:
            raise RuntimeError(f"{method} failed: {sol.message}")
        n_steps = sol.t.size - 1 if t_eval is None else 0
        return Solution(sol.t, sol.y.reshape(shape + (-1,)), calls, f"scipy {method}", n_steps)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def simulate(
    model: HodgkinHuxley,
    t_span: tuple[float, float],
    y0: ArrayLike | None = None,
    dt: float = SOLVER.rk4_dt,
    store_every: int = 1,
    method: str = "rk4",
    rtol: float = SOLVER.rtol,
    atol: float = SOLVER.atol,
    t_eval: ArrayLike | None = None,
    **solver_kw,
) -> Solution:
    """Integrate ``model``, restarting the solver at every stimulus breakpoint.

    The interval is split at the stimulus's ``breakpoints``. On each piece the
    current is frozen at its value at the piece's midpoint, so the ODE solved
    there is smooth and every method keeps its order of accuracy (a jump inside
    a step drops RK4 to first order; adaptive methods would waste rejected
    steps hunting for it).

    Args:
        model: the HH model; its stimulus may return per-trial arrays.
        t_span: (t0, t1) in ms.
        y0: initial state; defaults to the resting state (broadcast over trials
            if the stimulus is batched).
        dt: RK4 maximum step (ms); each piece uses the largest step <= dt that divides it.
        store_every: RK4 only: keep every k-th step.
        method: "rk4", "dopri45" (ours), or a scipy solve_ivp method name.
        rtol, atol: tolerances for adaptive methods.
        t_eval: output times for adaptive methods (dense output); default is every step.
        **solver_kw: forwarded to the underlying solver.
    """
    if t_eval is not None and method == "rk4":
        raise ValueError("RK4 outputs its own grid; use store_every instead of t_eval")
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
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)

    ts: list[FloatArray] = []
    ys: list[FloatArray] = []
    nfev = n_acc = n_rej = 0
    name = method
    for k, (a, b) in enumerate(pieces):
        last = k == len(pieces) - 1
        piece = replace(model, stimulus=ConstantCurrent(model.stimulus(0.5 * (a + b))))
        if t_eval is None:
            sol = _solve_piece(piece.rhs, (a, b), y, method, dt, rtol, atol, None, store_every, solver_kw)
            skip = 0 if not ts else 1  # first point of a piece repeats the previous piece's last
            ts.append(sol.t[skip:])
            ys.append(sol.y[..., skip:])
        else:
            wanted = t_eval[(t_eval >= a) & ((t_eval <= b) if last else (t_eval < b))]
            # always ask for b as well, to get the state that starts the next piece
            ask = wanted if wanted.size and wanted[-1] == b else np.append(wanted, b)
            sol = _solve_piece(piece.rhs, (a, b), y, method, dt, rtol, atol, ask, store_every, solver_kw)
            ts.append(sol.t[: wanted.size])
            ys.append(sol.y[..., : wanted.size])
        nfev += sol.nfev
        n_acc += sol.n_accepted
        n_rej += sol.n_rejected
        name = sol.method
        y = sol.y[..., -1]

    return Solution(np.concatenate(ts), np.concatenate(ys, axis=-1), nfev, name, n_acc, n_rej)


def membrane_derivative(model: HodgkinHuxley, t: ArrayLike, y: ArrayLike) -> FloatArray:
    """dV/dt at every stored point of a single-trial solution (t shape (n_t,), y shape (4, n_t))."""
    from hh.model import ionic_currents  # local import keeps the module namespace small

    t = np.asarray(t, dtype=float)
    I_ext = np.array([np.asarray(model.stimulus(ti), dtype=float) for ti in t])
    return (I_ext - np.sum(ionic_currents(y, model.params), axis=0)) / model.params.C_m


def spike_times(
    t: ArrayLike,
    V: ArrayLike,
    threshold: float = SPIKE_THRESHOLD_MV,
    dVdt: ArrayLike | None = None,
) -> FloatArray:
    """Times of upward crossings of ``threshold``.

    Without ``dVdt`` the crossing is located by linear interpolation between the two
    samples that bracket it: error O(dt^2), plenty for counting spikes and rates.

    With ``dVdt`` (the exact derivative at each sample, e.g. from
    :func:`membrane_derivative`) a cubic Hermite interpolant is built on the bracketing
    interval and its root refined by Newton's method: error O(dt^4), matching RK4, so
    spike-time errors measure the solver rather than the post-processing.

    Args:
        t: times, shape (n_t,).
        V: membrane potential for a single trial, shape (n_t,).
        dVdt: optional dV/dt at the same samples.
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    above = V >= threshold
    idx = np.flatnonzero(~above[:-1] & above[1:])  # V[i] < thr <= V[i+1]
    frac = (threshold - V[idx]) / (V[idx + 1] - V[idx])
    if dVdt is None or idx.size == 0:
        return t[idx] + frac * (t[idx + 1] - t[idx])

    dVdt = np.asarray(dVdt, dtype=float)
    h = t[idx + 1] - t[idx]
    p0, p1 = V[idx] - threshold, V[idx + 1] - threshold
    m0, m1 = dVdt[idx] * h, dVdt[idx + 1] * h
    s = frac.copy()
    for _ in range(8):  # Newton on the Hermite cubic H(s), s in [0, 1]
        h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
        H = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        dH = (6*s**2 - 6*s) * p0 + (3*s**2 - 4*s + 1) * m0 + (-6*s**2 + 6*s) * p1 + (3*s**2 - 2*s) * m1
        s = np.clip(s - H / np.where(dH == 0, 1.0, dH), 0.0, 1.0)
    return t[idx] + s * h


def fires(V: ArrayLike, threshold: float = SPIKE_THRESHOLD_MV) -> NDArray[np.bool_]:
    """Whether each trial contains at least one upward crossing of ``threshold``.

    Args:
        V: voltages with time on the last axis, shape (..., n_t).
    """
    V = np.asarray(V)
    return np.any((V[..., :-1] < threshold) & (V[..., 1:] >= threshold), axis=-1)


def bisect_threshold(
    responds: Callable[[FloatArray], NDArray[np.bool_]],
    lo: ArrayLike,
    hi: ArrayLike,
    rel_tol: float = 1e-6,
    max_iter: int = 60,
) -> FloatArray:
    """Vectorized bisection for the smallest stimulus amplitude that produces a spike.

    Each iteration halves every trial's bracket [lo, hi] at once, so one batched
    simulation per iteration serves all trials. Assumes ``responds`` is monotone in
    amplitude (False below threshold, True above). Trials whose ``hi`` doesn't fire
    get ``inf`` (no threshold up to ``hi``).

    Args:
        responds: maps amplitudes of shape (N,) to booleans of shape (N,).
        lo, hi: initial brackets, broadcastable to (N,).
    """
    lo, hi = np.broadcast_arrays(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    lo, hi = lo.copy(), hi.copy()
    ok_hi = responds(hi)
    if np.any(responds(lo)):
        raise ValueError("lower bracket already fires for some trials")
    for _ in range(max_iter):
        if np.all(hi - lo <= rel_tol * hi):
            break
        mid = 0.5 * (lo + hi)
        f = responds(mid)
        hi = np.where(f, mid, hi)
        lo = np.where(f, lo, mid)
    return np.where(ok_hi, hi, np.inf)
