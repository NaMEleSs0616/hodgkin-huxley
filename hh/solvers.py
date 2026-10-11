"""ODE solvers written from scratch in NumPy.

All solvers take a right-hand side ``f(t, y)`` and return a :class:`Solution`
with the same layout as scipy's ``solve_ivp``: ``t`` has shape ``(n_t,)`` and
``y`` has shape ``y0.shape + (n_t,)``, so ``sol.y[0]`` is always V(t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    """

    t: FloatArray
    y: FloatArray
    nfev: int
    method: str


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
    return Solution(t=t_out, y=np.moveaxis(ys, 0, -1), nfev=4 * n_steps, method="RK4")
