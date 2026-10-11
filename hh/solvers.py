"""ODE solvers written from scratch in NumPy.

* :func:`rk4`            classical fixed-step Runge-Kutta, written out by hand
* :func:`fixed_step_rk`  any explicit Butcher tableau with a fixed step (Euler, midpoint, RK4)
* :func:`dopri45`        adaptive Dormand-Prince 5(4) with error control, FSAL and dense output

All solvers take a right-hand side ``f(t, y)`` and return a :class:`Solution`
with the same layout as scipy's ``solve_ivp``: ``t`` has shape ``(n_t,)`` and
``y`` has shape ``y0.shape + (n_t,)``, so ``sol.y[0]`` is always V(t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
RHS = Callable[[float, FloatArray], FloatArray]


@dataclass
class Solution:
    """Result of an integration.

    Attributes:
        t: time points, shape (n_t,).
        y: states, shape y0.shape + (n_t,).
        nfev: number of right-hand-side evaluations (the cost measure we compare solvers by).
        method: name of the solver that produced it.
        n_accepted: accepted steps (adaptive solvers; equals the step count for fixed-step ones).
        n_rejected: rejected step attempts (adaptive solvers only).
    """

    t: FloatArray
    y: FloatArray
    nfev: int
    method: str
    n_accepted: int = 0
    n_rejected: int = 0


def rk4(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    dt: float,
    store_every: int = 1,
) -> Solution:
    """Classical fourth-order Runge-Kutta with a fixed step.

    One step from (t, y) with step size dt:

        k1 = f(t,        y)
        k2 = f(t + dt/2, y + dt/2 k1)
        k3 = f(t + dt/2, y + dt/2 k2)
        k4 = f(t + dt,   y + dt   k3)
        y_next = y + dt/6 (k1 + 2 k2 + 2 k3 + k4)

    The local truncation error is O(dt^5), so the global error over a fixed
    interval is O(dt^4).

    Args:
        f: right-hand side f(t, y).
        t_span: (t0, t1). (t1 - t0) must be an integer multiple of dt, so every
            step has the same size and the final time is hit exactly.
        y0: initial state, shape (d,) or (d, N) for N trials at once.
        dt: step size.
        store_every: keep every k-th step (saves memory on long runs). The final
            state is always stored.

    Returns:
        Solution with nfev = 4 * number of steps.
    """
    t0, t1 = map(float, t_span)
    n_steps_float = (t1 - t0) / dt
    n_steps = int(round(n_steps_float))
    if n_steps < 1 or abs(n_steps - n_steps_float) > 1e-9 * max(1.0, n_steps_float):
        raise ValueError(f"(t1 - t0) / dt = {n_steps_float} must be a positive integer")

    y = np.array(y0, dtype=float)
    stored_idx = list(range(0, n_steps + 1, store_every))
    if stored_idx[-1] != n_steps:
        stored_idx.append(n_steps)
    ys = np.empty((len(stored_idx),) + y.shape)
    ys[0] = y
    slot = 1

    half = 0.5 * dt
    for i in range(1, n_steps + 1):
        # t is recomputed from the step index instead of accumulating t += dt,
        # which would build up round-off error over many steps.
        t = t0 + (i - 1) * dt
        k1 = f(t, y)
        k2 = f(t + half, y + half * k1)
        k3 = f(t + half, y + half * k2)
        k4 = f(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if slot < len(stored_idx) and i == stored_idx[slot]:
            ys[slot] = y
            slot += 1

    t_out = t0 + dt * np.asarray(stored_idx, dtype=float)
    return Solution(t=t_out, y=np.moveaxis(ys, 0, -1), nfev=4 * n_steps, method="RK4",
                    n_accepted=n_steps)


def _check_fixed_steps(t_span: tuple[float, float], dt: float) -> tuple[float, float, int]:
    t0, t1 = map(float, t_span)
    n_float = (t1 - t0) / dt
    n = int(round(n_float))
    if n < 1 or abs(n - n_float) > 1e-9 * max(1.0, n_float):
        raise ValueError(f"(t1 - t0) / dt = {n_float} must be a positive integer")
    return t0, t1, n


# ---------------------------------------------------------------------------
# Butcher tableaux
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ButcherTableau:
    """An explicit Runge-Kutta method: stages k_i = f(t + c_i h, y + h sum_j A_ij k_j),
    update y_next = y + h sum_i b_i k_i."""

    name: str
    A: FloatArray
    b: FloatArray
    c: FloatArray
    order: int


EULER = ButcherTableau("Euler", np.array([[0.0]]), np.array([1.0]), np.array([0.0]), 1)
MIDPOINT = ButcherTableau(
    "Midpoint (RK2)", np.array([[0.0, 0.0], [0.5, 0.0]]), np.array([0.0, 1.0]), np.array([0.0, 0.5]), 2
)
RK4_TABLEAU = ButcherTableau(
    "RK4",
    np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], dtype=float),
    np.array([1, 2, 2, 1]) / 6.0,
    np.array([0, 0.5, 0.5, 1.0]),
    4,
)

# Dormand & Prince (1980). The 5th-order weights B5 propagate the solution
# ("local extrapolation"); the embedded 4th-order weights B4 only estimate the error.
_DP_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_DP_A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1 / 5, 0, 0, 0, 0, 0, 0],
    [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
    [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
])
_DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_DP_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_DP_E = _DP_B5 - _DP_B4
# Dense output (Shampine 1986): y(t + x h) = y + h sum_i k_i sum_j P_ij x^(j+1), 0 <= x <= 1.
_DP_P = np.array([
    [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
    [0, 0, 0, 0],
    [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
    [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
    [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
    [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
    [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
])
DOPRI5 = ButcherTableau("Dormand-Prince 5", _DP_A, _DP_B5, _DP_C, 5)


def stability_polynomial(tableau: ButcherTableau) -> FloatArray:
    """Coefficients [1, c_1, ..., c_s] (lowest degree first) of the stability function R(z).

    One step on y' = lambda y with z = h lambda gives the stage vector k = lambda (y 1 + h A k),
    so R(z) = 1 + z b^T (I - z A)^{-1} 1 = 1 + sum_{k>=1} z^k b^T A^{k-1} 1 (Neumann series).
    For an explicit method A is strictly lower triangular, hence nilpotent (A^s = 0),
    and the series stops: R is a polynomial of degree <= s.
    """
    s = len(tableau.b)
    coeffs = [1.0]
    v = np.ones(s)
    for _ in range(s):
        coeffs.append(float(tableau.b @ v))
        v = tableau.A @ v
    return np.trim_zeros(np.array(coeffs), "b")


def stability_function(tableau: ButcherTableau, z: ArrayLike) -> NDArray[np.complex128]:
    """R(z): one step on y' = lambda y maps y to R(h lambda) y. Stable iff |R(h lambda)| <= 1."""
    return np.polynomial.polynomial.polyval(np.asarray(z, dtype=complex), stability_polynomial(tableau))


def stable_step_limit(tableau: ButcherTableau, eigenvalues: ArrayLike, h_max: float = 1e3) -> float:
    """Largest h for which every h*lambda lies in the method's stability region.

    For each eigenvalue we walk outward along the ray {h lambda : h > 0} and bisect
    for the first point where |R| exceeds 1. Eigenvalues with Re >= 0 are skipped:
    those modes grow in the exact solution too, so stability can't require them to decay.
    """
    coeffs = stability_polynomial(tableau)
    R = lambda z: np.abs(np.polynomial.polynomial.polyval(z, coeffs))
    grid = np.geomspace(1e-6, h_max, 600)
    limit = h_max
    for lam in np.ravel(np.asarray(eigenvalues, dtype=complex)):
        if lam.real >= 0:
            continue
        bad = R(grid * lam) > 1 + 1e-12
        if not bad.any():
            continue
        i = int(np.argmax(bad))
        lo, hi = (grid[i - 1] if i else 0.0), grid[i]
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if R(mid * lam) > 1 + 1e-12 else (mid, hi)
        limit = min(limit, lo)
    return limit


def fixed_step_rk(
    f: RHS, t_span: tuple[float, float], y0: ArrayLike, dt: float, tableau: ButcherTableau
) -> Solution:
    """Any explicit Runge-Kutta method with a fixed step (general but slower than :func:`rk4`)."""
    t0, _, n_steps = _check_fixed_steps(t_span, dt)
    y = np.array(y0, dtype=float)
    ys = np.empty((n_steps + 1,) + y.shape)
    ys[0] = y
    s = len(tableau.b)
    k = np.empty((s,) + y.shape)
    for i in range(n_steps):
        t = t0 + i * dt
        for j in range(s):
            incr = np.tensordot(tableau.A[j, :j], k[:j], axes=1) if j else 0.0
            k[j] = f(t + tableau.c[j] * dt, y + dt * incr)
        y = y + dt * np.tensordot(tableau.b, k, axes=1)
        ys[i + 1] = y
    t_out = t0 + dt * np.arange(n_steps + 1)
    return Solution(t_out, np.moveaxis(ys, 0, -1), nfev=s * n_steps, method=tableau.name,
                    n_accepted=n_steps)


# ---------------------------------------------------------------------------
# Adaptive Dormand-Prince 5(4)
# ---------------------------------------------------------------------------
def _rms_norm(x: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _initial_step(f: RHS, t0: float, y0: FloatArray, f0: FloatArray, order: int,
                  rtol: float, atol: float) -> float:
    """Hairer, Norsett & Wanner's starting-step heuristic (Solving ODEs I, II.4).

    Guess h0 from the size of y and y', take one Euler step to estimate y'', then pick h so
    the leading error term h^(order+1) * max(|y'|, |y''|) is about 0.01 (in scaled units).
    """
    scale = atol + np.abs(y0) * rtol
    d0, d1 = _rms_norm(y0 / scale), _rms_norm(f0 / scale)
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    f1 = f(t0 + h0, y0 + h0 * f0)
    d2 = _rms_norm((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
    return min(100 * h0, h1)


def dopri45(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    t_eval: ArrayLike | None = None,
    h0: float | None = None,
    max_step: float = np.inf,
    controller: Literal["I", "PI"] = "I",
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 10.0,
) -> Solution:
    """Adaptive Dormand-Prince 5(4) Runge-Kutta.

    Each step computes 7 stages and two solutions: a 5th-order one (kept) and an
    embedded 4th-order one. Their difference estimates the local error of the
    4th-order solution, err = h * sum_i (b5_i - b4_i) k_i, which is O(h^5).

    Error control. Each component is scaled by sc_i = atol + rtol * max(|y_i|, |y_new_i|)
    and we take the RMS ||err / sc||. The step is accepted iff this is <= 1. Since the
    estimate scales like h^5, the step that would make it exactly 1 is
    h * (1/||err||)^(1/5); we aim a bit lower (``safety``) and clip the change to
    [min_factor, max_factor].

    * ``controller="I"`` uses exactly that rule (as scipy's RK45 does).
    * ``controller="PI"`` also uses the previous step's error,
      h_new = h * safety * err^(-0.17) * err_prev^(0.04)  (Hairer's DOPRI5 constants),
      which damps the oscillating accept/reject pattern seen when stability, not
      accuracy, limits the step.

    FSAL ("first same as last"). The last stage is evaluated at (t + h, y_new), which is
    exactly the first stage of the next step, so an accepted step costs 6 evaluations.

    Dense output. ``t_eval`` points are filled in with Shampine's 4th-order continuous
    extension, so output times don't force the step size.

    Returns:
        Solution at the accepted step points (or at ``t_eval``), with nfev,
        n_accepted and n_rejected.
    """
    t0, t1 = map(float, t_span)
    if t1 <= t0:
        raise ValueError("t_span must be increasing")
    y = np.array(y0, dtype=float)
    order_err = 4  # the error estimate is that of the 4th-order solution
    nfev = 0

    def fc(t: float, yy: FloatArray) -> FloatArray:
        nonlocal nfev
        nfev += 1
        return f(t, yy)

    fy = fc(t0, y)
    h = h0 if h0 is not None else _initial_step(fc, t0, y, fy, order_err, rtol, atol)
    h = min(h, max_step, t1 - t0)

    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        if np.any(np.diff(t_eval) < 0) or t_eval[0] < t0 or t_eval[-1] > t1:
            raise ValueError("t_eval must be sorted and inside t_span")
        out_y = np.empty((t_eval.size,) + y.shape)
        next_eval = 0
        while next_eval < t_eval.size and t_eval[next_eval] == t0:
            out_y[next_eval] = y
            next_eval += 1
    else:
        ts, ys = [t0], [y.copy()]

    K = np.empty((7,) + y.shape)
    t = t0
    err_prev = 1.0
    n_acc = n_rej = 0
    just_rejected = False
    min_step = 10 * np.finfo(float).eps

    while t < t1:
        if h < min_step * max(1.0, abs(t)):
            raise RuntimeError(f"step size underflow at t = {t}")
        t_new = t + h
        if t_new >= t1 or t1 - t_new < min_step * max(1.0, abs(t1)):
            t_new = t1  # land exactly on the end point
        h = t_new - t

        K[0] = fy
        # A step that is far too large can push a trial stage to absurd voltages where
        # exp() overflows. That step will be rejected anyway, so silence the warning
        # and treat a non-finite error estimate as "reject and shrink".
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(1, 7):
                incr = np.tensordot(_DP_A[i, :i], K[:i], axes=1)
                stage_y = y + h * incr
                K[i] = fc(t + _DP_C[i] * h, stage_y)
            y_new = stage_y  # row 7 of A equals b5, so the last stage point *is* y_new (FSAL)
            err_vec = h * np.tensordot(_DP_E, K, axes=1)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
            err = _rms_norm(err_vec / scale)
        if not np.isfinite(err):
            err = np.inf

        if err <= 1.0:
            if controller == "PI" and err > 0:
                factor = safety * err ** -0.17 * err_prev ** 0.04
            else:
                factor = max_factor if err == 0 else safety * err ** (-1.0 / (order_err + 1))
            factor = min(max_factor, max(min_factor, factor))
            if just_rejected:
                factor = min(1.0, factor)  # don't grow straight after a failure
            err_prev = max(err, 1e-4)

            if t_eval is not None:
                Q = np.tensordot(_DP_P.T, K, axes=1)  # shape (4,) + y.shape
                while next_eval < t_eval.size and t_eval[next_eval] <= t_new:
                    x = (t_eval[next_eval] - t) / h
                    powers = x ** np.arange(1, 5)
                    out_y[next_eval] = y + h * np.tensordot(powers, Q, axes=1)
                    next_eval += 1
            t, y, fy = t_new, y_new, K[6].copy()
            if t_eval is None:
                ts.append(t)
                ys.append(y.copy())
            n_acc += 1
            just_rejected = False
            h = min(h * factor, max_step)
        else:
            n_rej += 1
            just_rejected = True
            h *= min_factor if np.isinf(err) else max(min_factor, safety * err ** (-1.0 / (order_err + 1)))

    name = f"DOPRI5 (ours, {controller})"
    if t_eval is not None:
        if next_eval < t_eval.size:  # points exactly at t1 after round-off
            out_y[next_eval:] = y
        return Solution(t_eval, np.moveaxis(out_y, 0, -1), nfev, name, n_acc, n_rej)
    return Solution(np.array(ts), np.moveaxis(np.array(ys), 0, -1), nfev, name, n_acc, n_rej)
