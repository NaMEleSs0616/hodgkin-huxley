"""Simulation entry point, analysis helpers and experiment protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from hh.config import DEFAULT_PARAMS, SOLVER, SPIKE_THRESHOLD_MV, HHParameters
from hh.model import HodgkinHuxley
from hh.solvers import Solution, dopri45, rk4
from hh.stimuli import ConstantCurrent, StepCurrent

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

    The interval is split at the stimulus's ``breakpoints``, so no step straddles a
    jump (a jump inside a step drops RK4 to first order; adaptive methods would waste
    rejected steps hunting for it). For piecewise-constant stimuli the current is also
    frozen at its value at each piece's midpoint, which makes each piece autonomous and
    sidesteps the question of which side of a jump the endpoint stage should see.

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
        if getattr(model.stimulus, "piecewise_constant", False):
            piece = replace(model, stimulus=ConstantCurrent(model.stimulus(0.5 * (a + b))))
        else:
            piece = model  # smooth within the piece; evaluate the stimulus itself
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
    points: int = 15,
) -> FloatArray:
    """Vectorized multisection for the smallest stimulus amplitude that produces a spike.

    Plain bisection tests one midpoint per iteration and shrinks the bracket 2x. Here every
    iteration tests ``points`` equally spaced amplitudes inside each trial's bracket as one
    batched simulation and keeps the sub-interval where the response switches on, shrinking
    the bracket (points + 1)x. A batch of 15 costs little more than a batch of 1 (NumPy
    overhead dominates small arrays), so this is ~4x fewer simulations for the same precision.

    Assumes ``responds`` is monotone in amplitude (False below threshold, True above).
    Trials whose ``hi`` doesn't fire get ``inf`` (no threshold up to ``hi``).

    Args:
        responds: maps amplitudes of shape (N, K) to booleans of shape (N, K).
        lo, hi: initial brackets, broadcastable to (N,).
    """
    lo, hi = np.broadcast_arrays(np.atleast_1d(np.asarray(lo, dtype=float)),
                                 np.atleast_1d(np.asarray(hi, dtype=float)))
    lo, hi = lo.copy(), hi.copy()
    def checked(amps: FloatArray) -> NDArray[np.bool_]:
        r = np.asarray(responds(amps))
        if r.shape != amps.shape:
            raise ValueError(f"responds returned shape {r.shape} for amplitudes of shape {amps.shape}")
        return r

    ends = checked(np.stack([lo, hi], axis=1))
    if np.any(ends[:, 0]):
        raise ValueError("lower bracket already fires for some trials")
    ok_hi = ends[:, 1]
    frac = np.arange(1, points + 1) / (points + 1)
    rows = np.arange(lo.size)
    for _ in range(max_iter):
        active = ok_hi & (hi - lo > rel_tol * hi)
        if not active.any():
            break
        pts = lo[:, None] + (hi - lo)[:, None] * frac
        r = checked(pts)
        any_fire = r.any(axis=1)
        first = np.argmax(r, axis=1)  # first firing point (monotone response)
        new_hi = np.where(any_fire, pts[rows, first], hi)
        new_lo = np.where(any_fire, np.where(first > 0, pts[rows, np.maximum(first - 1, 0)], lo), pts[:, -1])
        hi = np.where(active, new_hi, hi)
        lo = np.where(active, new_lo, lo)
    return np.where(ok_hi, hi, np.inf)

# ---------------------------------------------------------------------------
# Equilibria
# ---------------------------------------------------------------------------
def equilibrium(params: HHParameters, I: float) -> FloatArray:
    """The (unique) equilibrium for constant current I: solve I_ss(V) = I, gates at x_inf(V).

    I_ss is strictly increasing for the HH parameters (checked in the tests), so there is
    exactly one equilibrium for every I and Brent's method on a wide bracket finds it.
    """
    from hh.model import GATES, steady_state, steady_state_current

    V = brentq(lambda v: steady_state_current(v, params) - I, -120.0, 60.0, xtol=1e-13)
    return np.array([V, *(float(steady_state(g, V)) for g in GATES)])


def equilibrium_max_real_eigenvalue(params: HHParameters, I: float) -> float:
    """Largest real part of the Jacobian's eigenvalues at the equilibrium (< 0: stable)."""
    y = equilibrium(params, I)
    J = HodgkinHuxley(params=params).jacobian(0.0, y)
    return float(np.linalg.eigvals(J).real.max())


# ---------------------------------------------------------------------------
# Spike-train statistics
# ---------------------------------------------------------------------------
def firing_rate(t: ArrayLike, V: ArrayLike, t_start: float) -> FloatArray:
    """Steady firing rate (Hz) per trial from spikes after ``t_start``.

    rate = 1000 * (n - 1) / (t_last - t_first), the reciprocal of the mean interspike
    interval; trials with fewer than two spikes have rate 0 (not repetitively firing).

    Args:
        V: shape (n_t,) or (N, n_t).
    """
    t = np.asarray(t)
    V2 = np.atleast_2d(np.asarray(V))
    keep = t >= t_start
    rates = np.zeros(V2.shape[0])
    for k, v in enumerate(V2):
        st = spike_times(t[keep], v[keep])
        if st.size >= 2:
            rates[k] = 1000.0 * (st.size - 1) / (st[-1] - st[0])
    return rates if np.ndim(V) > 1 else rates[0]


def ap_metrics(t: ArrayLike, V: ArrayLike, V_rest: float, after: float = 0.0) -> dict[str, float]:
    """Shape descriptors of the first action potential after time ``after``.

    Returns peak (mV), amplitude above rest (mV), half-width (ms, width at
    V_rest + amplitude/2), max upstroke slope dV/dt (mV/ms), AHP minimum (mV) and the
    AHP depth below rest (mV). NaNs if no spike.
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    nan = dict(peak=np.nan, amplitude=np.nan, half_width=np.nan, max_dvdt=np.nan,
               ahp_min=np.nan, ahp_depth=np.nan)
    sel = t >= after
    t, V = t[sel], V[sel]
    if not fires(V):
        return nan
    i_peak = int(np.argmax(V))
    peak = V[i_peak]
    half = V_rest + 0.5 * (peak - V_rest)
    up = spike_times(t[: i_peak + 1], V[: i_peak + 1], threshold=half)
    down = spike_times(t[i_peak:], -V[i_peak:], threshold=-half)
    width = (down[0] - up[-1]) if up.size and down.size else np.nan
    ahp = V[i_peak:].min()
    return dict(peak=peak, amplitude=peak - V_rest, half_width=width,
                max_dvdt=float(np.max(np.gradient(V, t))), ahp_min=ahp, ahp_depth=V_rest - ahp)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def pulse_threshold(
    params: HHParameters = DEFAULT_PARAMS,
    width: float = 0.5,
    y0: ArrayLike | None = None,
    hi: float = 100.0,
    window: float = 30.0,
    dt: float = SOLVER.rk4_dt,
    rel_tol: float = 1e-6,
    **sim_kw,
) -> FloatArray:
    """Smallest amplitude of a ``width``-ms current pulse that evokes a spike.

    The pulse starts at t = 0 from ``y0`` (default: the resting state); a spike is an
    upward 0 mV crossing within ``window`` ms. ``y0`` may hold one state per column to get
    one threshold per starting state in a single batched search. Returns ``inf`` where
    even ``hi`` fails. Extra keywords (e.g. ``method="dopri45"``) go to :func:`simulate`.
    """
    y0 = HodgkinHuxley(params=params).resting_state() if y0 is None else np.asarray(y0, dtype=float)

    def responds(amps: FloatArray) -> NDArray[np.bool_]:
        N, K = amps.shape
        start = np.repeat(y0, K, axis=1) if y0.ndim > 1 else y0
        model = HodgkinHuxley(params=params, stimulus=StepCurrent(amps.ravel(), t_on=0.0, t_off=width))
        sol = simulate(model, (0.0, window), start, dt=dt, store_every=2, **sim_kw)
        return fires(sol.y[0]).reshape(N, K)

    n = 1 if y0.ndim == 1 else y0.shape[1]
    out = bisect_threshold(responds, np.zeros(n), np.full(n, hi), rel_tol=rel_tol)
    return out if y0.ndim > 1 else out[0]


def step_threshold(
    params: HHParameters = DEFAULT_PARAMS,
    t_end: float = 100.0,
    t_check: float = 0.0,
    lo: float = 0.0,
    hi: float = 20.0,
    dt: float = SOLVER.rk4_dt,
    rel_tol: float = 1e-6,
    **sim_kw,
) -> float:
    """Smallest step current (switched on at t = 1 ms from rest) giving a spike after ``t_check``.

    ``t_check = 0`` gives the rheobase (any spike at all); a late ``t_check`` gives the onset
    of sustained repetitive firing.
    """

    def responds(amps: FloatArray) -> NDArray[np.bool_]:
        model = HodgkinHuxley(params=params, stimulus=StepCurrent(amps.ravel(), t_on=1.0))
        sol = simulate(model, (0.0, t_end), dt=dt, store_every=2, **sim_kw)
        return fires(sol.y[0][..., sol.t >= t_check]).reshape(amps.shape)

    return float(bisect_threshold(responds, lo, hi, rel_tol=rel_tol)[0])


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------
def oscillation_frequency(t: ArrayLike, V: ArrayLike, t_start: float, min_amplitude: float = 1.0) -> FloatArray:
    """Frequency (Hz) of sustained oscillations, spikes or not.

    Unlike :func:`firing_rate` (which counts 0 mV crossings, i.e. action potentials), this
    counts upward crossings of each trace's own midpoint (max + min)/2 after ``t_start``, so
    it also sees the small-amplitude limit cycle near depolarization block. Trials whose
    peak-to-peak range is below ``min_amplitude`` mV count as not oscillating.
    """
    t = np.asarray(t)
    V2 = np.atleast_2d(np.asarray(V))
    keep = t >= t_start
    out = np.zeros(V2.shape[0])
    for k, v in enumerate(V2[:, keep]):
        if v.max() - v.min() < min_amplitude:
            continue
        st = spike_times(t[keep], v, threshold=0.5 * (v.max() + v.min()))
        if st.size >= 2:
            out[k] = 1000.0 * (st.size - 1) / (st[-1] - st[0])
    return out if np.ndim(V) > 1 else out[0]


@dataclass
class FICurve:
    currents: FloatArray  # uA/cm^2
    spike_rate: FloatArray  # Hz, from 0 mV crossings
    oscillation_freq: FloatArray  # Hz, from midpoint crossings (any sustained oscillation)
    amplitude: FloatArray  # late-time peak-to-peak V (mV)


def fi_curve(
    currents: ArrayLike,
    params: HHParameters = DEFAULT_PARAMS,
    y0: ArrayLike | None = None,
    t_end: float = 600.0,
    t_transient: float = 200.0,
    dt: float = SOLVER.rk4_dt,
    store_every: int = 10,
) -> FICurve:
    """Steady firing statistics for constant currents, all trials in one batch.

    Args:
        currents: shape (N,), applied from t = 0.
        y0: shape (4,) (shared) or (4, N) (one start per current). Default: rest at I = 0,
            so switching the current on acts as a step from rest.
    """
    currents = np.asarray(currents, dtype=float)
    model = HodgkinHuxley(params=params, stimulus=ConstantCurrent(currents))
    sol = simulate(model, (0.0, t_end), y0, dt=dt, store_every=store_every)
    V = sol.y[0]
    late = sol.t >= t_transient
    return FICurve(currents, firing_rate(sol.t, V, t_transient),
                   oscillation_frequency(sol.t, V, t_transient),
                   V[:, late].max(axis=1) - V[:, late].min(axis=1))


def passive_ceiling(params: HHParameters, width: float, **kw) -> float:
    """Pulse amplitude at which a membrane *without* Na channels reaches 0 mV on its own.

    Above this amplitude a 0 mV crossing no longer proves a regenerative action potential:
    the injected charge alone gets there. Thresholds are only meaningful well below it.
    """
    return float(pulse_threshold(replace(params, g_Na=0.0), width, **kw))


def hopf_interval(params: HHParameters, currents: ArrayLike) -> tuple[float, float] | None:
    """(I_low, I_high) between which the equilibrium is unstable, refined by Brent's method.

    Returns None if the equilibrium is stable for every current in ``currents``.
    """
    currents = np.asarray(currents, dtype=float)
    f = lambda I: equilibrium_max_real_eigenvalue(params, I)
    vals = np.array([f(I) for I in currents])
    unstable = np.flatnonzero(vals > 0)
    if unstable.size == 0:
        return None
    i0, i1 = unstable[0], unstable[-1]
    lo = brentq(f, currents[i0 - 1], currents[i0], xtol=1e-10) if i0 > 0 else currents[0]
    hi = brentq(f, currents[i1], currents[i1 + 1], xtol=1e-10) if i1 + 1 < currents.size else currents[-1]
    return float(lo), float(hi)


@dataclass
class RefractoryResult:
    intervals: FloatArray  # ms, pulse onset to pulse onset
    thresholds: FloatArray  # uA/cm^2 (inf = no spike up to the cap)
    resting_threshold: float
    states: FloatArray  # (4, N) state at the second pulse's onset
    conditioning: Solution  # the single-pulse trajectory the states were read from
    absolute_end: float  # last interval with no response up to the cap
    relative_end: float  # first interval after which the threshold stays <= recovered_ratio x rest


def refractory_curve(
    params: HHParameters = DEFAULT_PARAMS,
    width: float = 0.5,
    conditioning_multiple: float = 2.0,
    t_first: float = 1.0,
    intervals: ArrayLike = np.arange(1.5, 40.0, 0.1),
    cap_multiple: float = 10.0,
    recovered_ratio: float = 1.05,
    window_after: float = 15.0,
    dt: float = 0.005,
    rel_tol: float = 1e-4,
) -> RefractoryResult:
    """Second-pulse threshold as a function of the interval after a conditioning spike.

    Uses the flow (semigroup) property of the ODE: the state at the second pulse's onset
    depends only on the interval, so the conditioning spike is simulated once and each
    trial starts from the stored state. Only the ``window_after`` ms after the second
    pulse are then simulated, for all intervals at once.

    Intervals that start before the conditioning spike's upstroke has crossed 0 mV are
    dropped, since that crossing would be mistaken for a response to the second pulse.
    """
    th_rest = float(pulse_threshold(params, width, dt=dt, rel_tol=1e-9))
    t_end = t_first + float(np.max(intervals)) + 1.0
    cond = simulate(
        HodgkinHuxley(params=params,
                      stimulus=StepCurrent(conditioning_multiple * th_rest, t_first, t_first + width)),
        (0.0, t_end), dt=dt)
    first_cross = spike_times(cond.t, cond.y[0])[0]
    intervals = np.asarray(intervals, dtype=float)
    intervals = intervals[t_first + intervals > first_cross]
    # read the stored states exactly (intervals are multiples of dt)
    idx = np.searchsorted(np.round(cond.t, 9), np.round(t_first + intervals, 9))
    if not np.allclose(cond.t[idx], t_first + intervals):
        raise ValueError("intervals must be multiples of dt")
    states = cond.y[:, idx]

    th = pulse_threshold(params, width, states, hi=cap_multiple * th_rest, window=window_after,
                         dt=dt, rel_tol=rel_tol)
    ratio = th / th_rest
    blocked = np.isinf(th)
    absolute_end = float(intervals[blocked].max()) if blocked.any() else float("nan")
    not_recovered = np.flatnonzero(ratio > recovered_ratio)
    relative_end = float(intervals[not_recovered.max() + 1]) if not_recovered.size else float("nan")
    return RefractoryResult(intervals, th, th_rest, states, cond, absolute_end, relative_end)


def fit_h_n_relation(params: HHParameters = DEFAULT_PARAMS, I: float = 10.0,
                     t_end: float = 200.0, t_transient: float = 100.0) -> tuple[float, float, float]:
    """Least-squares fit h ~ h0 - k n on the full model's limit cycle.

    Returns (h0, k, rms residual).
    """
    sol = simulate(HodgkinHuxley(params=params, stimulus=ConstantCurrent(I)), (0.0, t_end), dt=0.01)
    late = sol.t >= t_transient
    n, h = sol.y[2, late], sol.y[3, late]
    A = np.column_stack([np.ones_like(n), -n])
    (h0, k), *_ = np.linalg.lstsq(A, h, rcond=None)
    resid = h - (h0 - k * n)
    return float(h0), float(k), float(np.sqrt(np.mean(resid**2)))
