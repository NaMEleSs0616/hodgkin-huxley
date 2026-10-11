"""Injected-current protocols I_ext(t), in uA/cm^2.

Each stimulus is a callable ``I(t)``. Amplitudes may be NumPy arrays: then
``I(t)`` returns one current per trial, which lets a single integration run a
whole batch of experiments at once (state shape ``(4, N)``).

Stimuli expose ``breakpoints``: the times where I_ext (or its slope) jumps.
``hh.experiments.simulate`` integrates each piece separately, because a jump
inside a step destroys a Runge-Kutta method's order of accuracy (RK4 drops from
4th to 1st order on a current step). Stimuli with ``piecewise_constant = True``
are additionally frozen to a constant on each piece.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

Current = Union[float, NDArray[np.float64]]
Stimulus = Callable[[float], Current]


def no_current(t: float) -> float:
    """I_ext = 0."""
    return 0.0


@dataclass(frozen=True)
class ConstantCurrent:
    """I_ext = ``amplitude`` for all t."""

    amplitude: ArrayLike
    piecewise_constant: ClassVar[bool] = True

    def __call__(self, t: float) -> Current:
        return np.asarray(self.amplitude, dtype=float)

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return ()


@dataclass(frozen=True)
class StepCurrent:
    """Constant current ``amplitude`` switched on for ``t_on <= t < t_off``."""

    amplitude: ArrayLike
    t_on: float = 0.0
    t_off: float = np.inf
    piecewise_constant: ClassVar[bool] = True

    def __call__(self, t: float) -> Current:
        on = self.t_on <= t < self.t_off
        return np.asarray(self.amplitude, dtype=float) * on

    @property
    def breakpoints(self) -> tuple[float, ...]:
        """Times where I_ext jumps (adaptive solvers should restart here)."""
        return tuple(b for b in (self.t_on, self.t_off) if np.isfinite(b))


@dataclass(frozen=True)
class PulsePair:
    """Two rectangular pulses of equal width; the second starts ``interval`` ms after the first.

    ``amplitude`` and ``interval`` broadcast against each other, so either can be an array.
    """

    amplitude: ArrayLike
    width: float
    t_first: float
    interval: ArrayLike
    piecewise_constant: ClassVar[bool] = True

    def __call__(self, t: float) -> Current:
        amp = np.asarray(self.amplitude, dtype=float)
        t2 = self.t_first + np.asarray(self.interval, dtype=float)
        first = self.t_first <= t < self.t_first + self.width
        second = (t2 <= t) & (t < t2 + self.width)
        return amp * (first | second)

    @property
    def breakpoints(self) -> tuple[float, ...]:
        t2 = self.t_first + np.ravel(self.interval).astype(float)
        edges = np.concatenate([[self.t_first, self.t_first + self.width], t2, t2 + self.width])
        return tuple(np.unique(edges))


@dataclass(frozen=True)
class RampCurrent:
    """Current ramping linearly from ``I_start`` (before ``t_start``) to ``I_end`` (after ``t_end``).

    Continuous but with kinks at t_start and t_end, which are its breakpoints.
    """

    I_start: float
    I_end: float
    t_start: float
    t_end: float
    piecewise_constant: ClassVar[bool] = False

    def __call__(self, t: float) -> Current:
        frac = min(max((t - self.t_start) / (self.t_end - self.t_start), 0.0), 1.0)
        return self.I_start + frac * (self.I_end - self.I_start)

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return (self.t_start, self.t_end)
