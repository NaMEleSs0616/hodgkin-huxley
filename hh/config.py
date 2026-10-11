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


@dataclass(frozen=True)
class SpikingConfig:
    """Experiment 1: step currents, threshold, all-or-none (scripts/exp1_spiking.py)."""

    step_amplitudes: tuple[float, ...] = (1.5, 2.5, 5.0, 8.0, 20.0)  # uA/cm^2
    step_t_on: float = 10.0
    step_t_off: float = 110.0
    step_t_end: float = 130.0
    pulse_width: float = 0.5  # ms; brief compared with tau_n, tau_h so it acts like a charge kick
    pulse_t_on: float = 1.0
    pulse_window: float = 30.0  # ms to wait for a spike after the pulse
    threshold_hi: float = 100.0  # uA/cm^2 upper bisection bracket
    threshold_rel_tol: float = 1e-12  # the response is graded within ~1e-10 of threshold
    amplitude_scan: tuple[float, float, int] = (0.0, 30.0, 601)  # linspace for peak-V vs amplitude
    log_deltas: tuple[float, ...] = tuple(10.0**-k for k in range(2, 13))
    rheobase_t_end: float = 100.0
    repetitive_t_end: float = 600.0
    repetitive_t_check: float = 300.0  # spikes after this time count as sustained firing
    dt: float = 0.005


@dataclass(frozen=True)
class FICurveConfig:
    """Experiment 2: firing rate vs current (scripts/exp2_fi_curve.py)."""

    currents_fine: tuple[float, float, float] = (0.0, 20.0, 0.1)  # arange (start, stop, step)
    currents_coarse: tuple[float, float, float] = (20.0, 200.0, 2.0)
    t_end: float = 600.0
    t_transient: float = 200.0  # discard; rate is measured on [t_transient, t_end]
    store_every: int = 10  # keep V every 0.1 ms (ample for counting spikes)
    dt: float = 0.01
    # Starting exactly on an unstable equilibrium you never leave it (RK4 maps it to itself
    # bit for bit), so "from equilibrium" runs get a tiny seeded random kick, standing in for
    # channel noise. Growth near the Hopf point is slow (Re lambda ~ 0.004/ms), hence the long run.
    equilibrium_kick_mV: float = 0.1
    equilibrium_currents: tuple[float, float, float] = (0.0, 20.0, 0.1)
    equilibrium_t_end: float = 2000.0
    # Slow ramps reveal hysteresis: firing starts and stops at different currents.
    ramp_low: float = 4.0
    ramp_high: float = 20.0
    ramp_duration: float = 8000.0  # ms, i.e. 0.002 uA/cm^2 per ms (~0.03 per spike)
    ramp_rtol: float = 1e-7


@dataclass(frozen=True)
class RefractoryConfig:
    """Experiment 3: two-pulse protocol (scripts/exp3_refractory.py)."""

    pulse_width: float = 0.5
    conditioning_multiple: float = 2.0  # first pulse = 2x resting threshold, so it always fires
    t_first: float = 1.0
    intervals: tuple[float, float, float] = (1.5, 40.0, 0.1)  # onset-to-onset, ms (arange)
    cap_multiple: float = 10.0  # search second-pulse amplitudes up to 10x resting threshold
    recovered_ratio: float = 1.05  # relative refractory period ends when threshold <= 1.05x rest
    window_after: float = 15.0
    example_intervals: tuple[float, ...] = (4.0, 7.0, 10.0, 15.0, 20.0)
    example_multiple: float = 4.0
    rel_tol: float = 1e-4
    dt: float = 0.005


@dataclass(frozen=True)
class BlockerConfig:
    """Experiment 4: TTX (g_Na down) and TEA (g_K down) (scripts/exp4_blockers.py)."""

    ttx_fractions: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)  # remaining g_Na fraction
    # Rest stays stable only for g_K fraction > ~0.55 (see the bifurcation panel), so AP shapes
    # and f-I curves use fractions in that range; lower fractions are shown in the diagram.
    tea_fractions: tuple[float, ...] = (1.0, 0.85, 0.7, 0.6)
    pulse_width: float = 0.5
    shape_multiple: float = 1.5  # AP shapes evoked by a pulse 1.5x that condition's own threshold
    t_shape: float = 40.0
    threshold_hi: float = 400.0
    threshold_window: float = 40.0
    block_ratio: float = 0.9  # "blocked" once threshold >= 90 % of the passive ceiling
    ttx_scan: tuple[float, float, int] = (0.02, 1.0, 25)  # linspace of g_Na fractions
    tea_scan: tuple[float, float, int] = (0.0, 1.0, 101)  # linspace of g_K fractions
    tea_kick_mV: float = 0.1
    tea_t_end: float = 800.0
    tea_t_transient: float = 400.0
    fi_currents: tuple[float, float, float] = (0.0, 60.5, 0.5)
    fi_t_end: float = 400.0
    fi_t_transient: float = 150.0
    dt: float = 0.01


@dataclass(frozen=True)
class PhasePlaneConfig:
    """Experiment 5: reduced V-n plane (scripts/exp5_phase_plane.py)."""

    currents: tuple[float, ...] = (0.0, 10.0)
    V_range: tuple[float, float] = (-85.0, 55.0)
    n_range: tuple[float, float] = (0.25, 0.85)
    grid: int = 400
    # initial V at resting n (m at m_inf(V)); the kick threshold is ~ -60.1 mV in both models
    kick_voltages: tuple[float, ...] = (-62.0, -61.0, -59.0, -55.0, -40.0)
    t_end: float = 60.0
    fit_current: float = 10.0  # h ~ a - b n is fitted on the full model's limit cycle here


@dataclass(frozen=True)
class TemperatureConfig:
    """Experiment 6: Q10 temperature scaling (scripts/exp6_temperature.py)."""

    temperatures: tuple[float, ...] = (6.3, 10.0, 15.0, 20.0, 25.0, 30.0)
    q10: float = 3.0
    fi_temperatures: tuple[float, ...] = (6.3, 15.0, 25.0)
    fi_currents: tuple[float, float, float] = (0.0, 100.0, 1.0)
    block_search: tuple[float, float] = (6.3, 45.0)  # bracket for the heat-block temperature
    pulse_width: float = 0.5
    threshold_hi: float = 400.0
    base_dt: float = 0.01  # divided by phi: faster kinetics need proportionally smaller steps


SPIKING = SpikingConfig()
FI_CURVE = FICurveConfig()
REFRACTORY = RefractoryConfig()
BLOCKERS = BlockerConfig()
PHASE_PLANE = PhasePlaneConfig()
TEMPERATURE = TemperatureConfig()
