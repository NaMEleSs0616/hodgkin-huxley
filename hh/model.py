r"""The Hodgkin-Huxley equations.

State vector y = (V, m, n, h):

    C dV/dt = I_ext - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L)
    dx/dt   = alpha_x(V) (1 - x) - beta_x(V) x,        x in {m, n, h}

Every function here broadcasts: V may be a scalar or an array, and the state
may have shape ``(4,)`` or ``(4, N)`` for N independent trials.

Removable singularities
-----------------------
alpha_m and alpha_n have the form ``a (V - V0) / (1 - exp(-(V - V0)/k))``, which is
0/0 at V = V0. Substituting u = -(V - V0)/k gives

    a (V - V0) / (1 - e^{-(V-V0)/k}) = a k * u / (e^u - 1) = a k / exprel(u),

where exprel(u) = (e^u - 1)/u, with exprel(0) = 1 by continuity. scipy.special.exprel
evaluates this accurately for all u (a Taylor series near 0), so the code never
divides 0 by 0 and never loses digits to cancellation near the singular voltage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import exprel

from hh.config import DEFAULT_PARAMS, HHParameters
from hh.stimuli import Stimulus, no_current

FloatArray = NDArray[np.float64]

STATE_NAMES: tuple[str, ...] = ("V", "m", "n", "h")
GATES: tuple[str, ...] = ("m", "n", "h")


# ---------------------------------------------------------------------------
# Rate functions (1/ms), at the reference temperature
# ---------------------------------------------------------------------------
def alpha_m(V: ArrayLike) -> FloatArray:
    """0.1 (V+40) / (1 - exp(-(V+40)/10)), written as 1/exprel(-(V+40)/10)."""
    return 1.0 / exprel(-(np.asarray(V) + 40.0) / 10.0)


def beta_m(V: ArrayLike) -> FloatArray:
    return 4.0 * np.exp(-(np.asarray(V) + 65.0) / 18.0)


def alpha_n(V: ArrayLike) -> FloatArray:
    """0.01 (V+55) / (1 - exp(-(V+55)/10)), written as 0.1/exprel(-(V+55)/10)."""
    return 0.1 / exprel(-(np.asarray(V) + 55.0) / 10.0)


def beta_n(V: ArrayLike) -> FloatArray:
    return 0.125 * np.exp(-(np.asarray(V) + 65.0) / 80.0)


def alpha_h(V: ArrayLike) -> FloatArray:
    return 0.07 * np.exp(-(np.asarray(V) + 65.0) / 20.0)


def beta_h(V: ArrayLike) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(V) + 35.0) / 10.0))


RATE_FUNCTIONS = {
    "m": (alpha_m, beta_m),
    "n": (alpha_n, beta_n),
    "h": (alpha_h, beta_h),
}


def rates(gate: str, V: ArrayLike, phi: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """(alpha, beta) for one gate, scaled by the temperature factor phi."""
    a, b = RATE_FUNCTIONS[gate]
    return phi * a(V), phi * b(V)


def steady_state(gate: str, V: ArrayLike) -> FloatArray:
    """x_inf(V) = alpha / (alpha + beta): the value x relaxes to if V is held fixed.

    Independent of temperature, because phi cancels in the ratio.
    """
    a, b = rates(gate, V)
    return a / (a + b)


def time_constant(gate: str, V: ArrayLike, phi: float = 1.0) -> FloatArray:
    """tau_x(V) = 1 / (alpha + beta), in ms. Then dx/dt = (x_inf - x) / tau_x."""
    a, b = rates(gate, V, phi)
    return 1.0 / (a + b)


# ---------------------------------------------------------------------------
# Currents and the vector field
# ---------------------------------------------------------------------------
def ionic_currents(
    y: ArrayLike, params: HHParameters = DEFAULT_PARAMS
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(I_Na, I_K, I_L) in uA/cm^2, positive = outward (the electrophysiology sign convention)."""
    V, m, n, h = np.asarray(y, dtype=float)
    p = params
    I_Na = p.g_Na * m**3 * h * (V - p.E_Na)
    I_K = p.g_K * n**4 * (V - p.E_K)
    I_L = p.g_L * (V - p.E_L)
    return I_Na, I_K, I_L


def steady_state_current(V: ArrayLike, params: HHParameters = DEFAULT_PARAMS) -> FloatArray:
    """Total ionic current with every gate at its steady state: I_ss(V).

    Equilibria of the full system satisfy I_ext = I_ss(V).
    """
    V = np.asarray(V, dtype=float)
    y = np.stack([V, *(steady_state(g, V) for g in GATES)])
    return sum(ionic_currents(y, params))


@dataclass
class HodgkinHuxley:
    """The HH vector field for one parameter set and one stimulus protocol.

    ``model.rhs(t, y)`` has the (t, y) signature that both our solvers and
    scipy.integrate.solve_ivp expect.
    """

    params: HHParameters = DEFAULT_PARAMS
    stimulus: Stimulus = field(default=no_current)

    def rhs(self, t: float, y: FloatArray) -> FloatArray:
        """dy/dt for state y of shape (4,) or (4, N)."""
        p = self.params
        V, m, n, h = y
        I_Na, I_K, I_L = ionic_currents(y, p)
        I_ext = self.stimulus(t)

        dV = (I_ext - I_Na - I_K - I_L) / p.C_m
        if np.shape(dV) != np.shape(V):
            raise ValueError(
                f"stimulus returned shape {np.shape(I_ext)} but the state has trial shape "
                f"{np.shape(V)}; give y0 one column per trial (hh.experiments.simulate does this)"
            )
        out = np.empty((4,) + np.shape(dV))  # dV already has the broadcast trial shape
        out[0] = dV
        for i, (gate, x) in enumerate(zip(GATES, (m, n, h)), start=1):
            a, b = rates(gate, V, p.phi)
            out[i] = a * (1.0 - x) - b * x
        return out

    def resting_potential(self, I_hold: float = 0.0) -> float:
        """Solve I_ss(V) = I_hold for V on the physiological branch.

        I_ss is monotone increasing on [-90, -40] mV for standard HH, so the root
        is unique there and bisection (Brent's method) is guaranteed to find it.
        """
        return float(brentq(lambda V: steady_state_current(V, self.params) - I_hold, -90.0, -40.0,
                            xtol=1e-13))

    def resting_state(self, I_hold: float = 0.0) -> FloatArray:
        """(V_rest, m_inf, n_inf, h_inf): the initial condition for every experiment."""
        V = self.resting_potential(I_hold)
        return np.array([V, *(float(steady_state(g, V)) for g in GATES)])
