"""Single source of truth for model parameters, numerical settings and plot style.

Every script and test imports from here, so changing a number in this file
changes it everywhere. Units throughout the package:

    time            ms
    voltage         mV
    current density uA/cm^2
    conductance     mS/cm^2
    capacitance     uF/cm^2

These units are self-consistent: (mS/cm^2) * mV = uA/cm^2 and
(uA/cm^2) / (uF/cm^2) = mV/ms, so no conversion factors appear in the equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths and reproducibility
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

RANDOM_SEED: int = 20261011


def make_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Return a seeded NumPy generator (used only by optional noisy-input extensions)."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HHParameters:
    """Hodgkin-Huxley squid-axon parameters (modern convention, rest near -65 mV).

    Frozen so a parameter set can't be mutated by accident; derive variants with
    ``dataclasses.replace`` (e.g. a TTX-treated axon) or the helpers below.
    """

    C_m: float = 1.0  # uF/cm^2
    g_Na: float = 120.0  # mS/cm^2
    g_K: float = 36.0  # mS/cm^2
    g_L: float = 0.3  # mS/cm^2
    E_Na: float = 50.0  # mV
    E_K: float = -77.0  # mV
    E_L: float = -54.387  # mV
    temperature: float = 6.3  # degC, the temperature HH's rate functions were fitted at
    reference_temperature: float = 6.3  # degC
    q10: float = 3.0  # rate multiplier per 10 degC

    @property
    def phi(self) -> float:
        """Temperature factor multiplying every alpha and beta: Q10^((T - T_ref)/10)."""
        return self.q10 ** ((self.temperature - self.reference_temperature) / 10.0)


DEFAULT_PARAMS = HHParameters()


# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverSettings:
    """Default step sizes and tolerances."""

    rk4_dt: float = 0.01  # ms; comfortably resolves a ~1 ms spike upstroke
    rtol: float = 1e-6
    atol: float = 1e-8
    reference_rtol: float = 1e-12  # for the "ground truth" solution in convergence studies
    reference_atol: float = 1e-12


SOLVER = SolverSettings()

SPIKE_THRESHOLD_MV: float = 0.0  # a spike is an upward crossing of this voltage


# ---------------------------------------------------------------------------
# Plot style (applied by hh.plotting.apply_style)
# ---------------------------------------------------------------------------
PLOT_RC: dict[str, object] = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
}

# One fixed color per state variable so V/m/n/h look the same in every figure.
STATE_COLORS: dict[str, str] = {
    "V": "#1f3a5f",
    "m": "#d1495b",
    "n": "#2a9d8f",
    "h": "#e9a03b",
}


# ---------------------------------------------------------------------------
# Experiment protocols (every number a script uses lives here)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationConfig:
    """Milestone 3: solver validation (scripts/solver_validation.py)."""

    I_drive: float = 10.0  # uA/cm^2, constant from t = 0 so the problem is smooth
    t_convergence: float = 20.0  # ms: covers one full action potential
    convergence_dts: tuple[float, ...] = tuple(0.04 / 2**k for k in range(7))  # 0.04 ... 0.000625
    t_work_precision: float = 100.0  # ms: ~7 spikes
    t_quiescent: float = 2000.0  # ms of subthreshold dynamics
    I_quiescent: float = 1.0  # uA/cm^2 step: below the single-spike threshold
    t_on_quiescent: float = 10.0
    tolerances: tuple[float, ...] = tuple(10.0**-k for k in range(3, 12))
    # Work-precision error = max |V - V_ref| on a 1 ms output grid (the whole trajectory, not
    # just the end: in the quiescent run every stable method ends at the same equilibrium).
    # RK4 steps are powers of 2 so they land exactly on that grid.
    error_grid_spacing: float = 1.0
    rk4_dts_spiking: tuple[float, ...] = tuple(2.0**-k for k in range(4, 10))  # 0.0625 ... 0.002
    rk4_dts_quiescent: tuple[float, ...] = tuple(2.0**-k for k in range(1, 7))  # 0.5 ... 0.016
    failure_dts: tuple[float, ...] = (0.01, 0.05, 0.08, 0.094, 0.1)
    failure_scan_dts: tuple[float, ...] = tuple(np.round(np.arange(0.005, 0.1001, 0.0025), 4))
    reference_method: str = "DOP853"
    # DOP853's dense output is ~1e-8 off between long steps (~1.3 ms in quiet stretches) even
    # when its step values are 1e-12 accurate, so the reference run caps the step.
    reference_max_step: float = 0.25
    stiffness_tols: tuple[float, ...] = (1e-3, 1e-6)
    stiffness_t_end: float = 60.0


VALIDATION = ValidationConfig()
