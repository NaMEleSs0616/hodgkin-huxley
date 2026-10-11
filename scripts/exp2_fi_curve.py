"""Experiment 2: firing rate vs injected current, bistability and the onset of repetitive firing.

Protocols:
    step from rest      every current switched on at t = 0 from the I = 0 resting state (batched)
    from equilibrium    start at the equilibrium for that current plus a tiny seeded random kick
    slow ramps          current swept 4 -> 12 -> 4 uA/cm^2: firing starts and stops at different currents

Figure:
    figures/exp2_fi_curve.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from hh import DEFAULT_PARAMS, HodgkinHuxley
from hh.config import FI_CURVE as CFG, STATE_COLORS, make_rng
from hh.experiments import (
    equilibrium,
    equilibrium_max_real_eigenvalue,
    fi_curve,
    hopf_interval,
    simulate,
    spike_times,
    step_threshold,
)
from hh.plotting import apply_style, save_figure
from hh.stimuli import RampCurrent


def ramp(I_from: float, I_to: float, y0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous rate (1000/ISI) vs the current at which each interval ended."""
    stim = RampCurrent(I_from, I_to, t_start=0.0, t_end=CFG.ramp_duration)
    sol = simulate(HodgkinHuxley(stimulus=stim), (0.0, CFG.ramp_duration), y0, method="dopri45",
                   rtol=CFG.ramp_rtol, atol=CFG.ramp_rtol)
    st = spike_times(sol.t, sol.y[0])
    if st.size < 2:
        return np.array([]), np.array([])
    return np.array([stim(t) for t in st[1:]]), 1000.0 / np.diff(st)


def main() -> None:
    rng = make_rng()
    currents = np.concatenate([np.arange(*CFG.currents_fine), np.arange(*CFG.currents_coarse),
                               [CFG.currents_coarse[1]]])
    step = fi_curve(currents, t_end=CFG.t_end, t_transient=CFG.t_transient, dt=CFG.dt,
                    store_every=CFG.store_every)

    I_eq = np.arange(*CFG.equilibrium_currents)
    y_eq = np.column_stack([equilibrium(DEFAULT_PARAMS, I) for I in I_eq])
    y_eq[0] += CFG.equilibrium_kick_mV * rng.standard_normal(I_eq.size)
    from_eq = fi_curve(I_eq, y0=y_eq, t_end=CFG.equilibrium_t_end,
                       t_transient=CFG.equilibrium_t_end - (CFG.t_end - CFG.t_transient),
                       dt=CFG.dt, store_every=2 * CFG.store_every)

    hopf = hopf_interval(DEFAULT_PARAMS, np.arange(0.0, 200.0, 0.5))
    onset_step = step_threshold(t_end=CFG.t_end, t_check=CFG.t_end - 300.0, lo=2.0, hi=15.0, dt=CFG.dt,
                                rel_tol=1e-6)
    on_eq = I_eq[from_eq.spike_rate > 0].min()
    print(f"equilibrium unstable (Hopf bifurcations) for {hopf[0]:.4f} < I < {hopf[1]:.4f} uA/cm^2")
    print(f"sustained firing after a step from rest starts at I = {onset_step:.4f} uA/cm^2")
    print(f"sustained firing from (kicked) equilibrium starts at I = {on_eq:.1f} uA/cm^2")

    # hysteresis by slow ramps: up from (kicked) rest, down from the end state of the up ramp
    y_low = equilibrium(DEFAULT_PARAMS, CFG.ramp_low)
    y_low[0] += CFG.equilibrium_kick_mV
    I_up, f_up = ramp(CFG.ramp_low, CFG.ramp_high, y_low)
    y_high = simulate(HodgkinHuxley(stimulus=RampCurrent(CFG.ramp_low, CFG.ramp_high, 0.0, CFG.ramp_duration)),
                      (0.0, CFG.ramp_duration), y_low, method="dopri45", rtol=CFG.ramp_rtol,
                      atol=CFG.ramp_rtol).y[:, -1]
    I_down, f_down = ramp(CFG.ramp_high, CFG.ramp_low, y_high)
    print(f"ramp up:   first interspike interval ends at I = {I_up.min():.3f} "
          f"(delayed past the Hopf point {hopf[0]:.2f} by slow passage)")
    print(f"ramp down: last interspike interval ends at I = {I_down.min():.3f} (fold of limit cycles)")
    print(f"=> bistable range ~({max(onset_step, I_down.min()):.2f}, {hopf[0]:.2f}): rest and "
          f"repetitive firing coexist")

    jump = step.spike_rate[step.spike_rate > 0][0]
    print(f"firing rate jumps from 0 to {jump:.1f} Hz at onset (type II excitability)")
    print(f"spikes (0 mV crossings) stop above ~{currents[step.spike_rate > 0].max():.0f} uA/cm^2; "
          f"small oscillations continue to ~{currents[step.oscillation_freq > 0].max():.0f} "
          f"(upper Hopf {hopf[1]:.1f})")
    for I in (6.3, 8.0, 10.0, 20.0, 50.0, 100.0):
        i = np.argmin(np.abs(currents - I))
        print(f"  I = {currents[i]:6.1f}: spikes {step.spike_rate[i]:6.1f} Hz, "
              f"oscillation {step.oscillation_freq[i]:6.1f} Hz, amplitude {step.amplitude[i]:6.1f} mV")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    ax = axes[0]
    ax.axvspan(*hopf, color="#e9a03b", alpha=0.12, lw=0, label="rest unstable (between Hopf points)")
    ax.plot(currents, step.spike_rate, color=STATE_COLORS["V"], label="spikes (0 mV crossings)")
    ax.plot(currents, step.oscillation_freq, "--", color="0.35", lw=1, label="oscillation frequency")
    ax.set_xlabel(r"injected current I ($\mu$A/cm$^2$)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("f-I curve (step from rest)")
    ax.legend(fontsize=7, loc="lower center")

    ax = axes[1]
    ax.axvspan(max(onset_step, I_down.min()), hopf[0], color="#d1495b", alpha=0.12, lw=0, label="bistable")
    ax.plot(currents, step.spike_rate, ".", ms=4, color=STATE_COLORS["V"], label="step from rest")
    ax.plot(I_eq, from_eq.spike_rate, ".", ms=4, color="#e9a03b", label="from kicked equilibrium")
    # each ramp drawn as a full path: silent stretch (rate 0), jump, firing branch
    up_I = np.concatenate([[CFG.ramp_low, I_up.min(), I_up.min()], I_up])
    up_f = np.concatenate([[3.0, 3.0, f_up[0]], f_up])
    order = np.argsort(I_down)
    dn_I = np.concatenate([[CFG.ramp_low, I_down.min()], I_down[order]])
    dn_f = np.concatenate([[-3.0, -3.0], f_down[order]])
    ax.plot(up_I, up_f, "-", color="#2a9d8f", lw=2, label="slow ramp up (from rest)")
    ax.plot(dn_I, dn_f, "-", color="#d1495b", lw=2, label="slow ramp down (from firing)")
    ax.annotate("", xy=(I_up.min(), f_up[0] - 3), xytext=(I_up.min(), 8),
                arrowprops=dict(arrowstyle="->", color="#2a9d8f"))
    ax.annotate("", xy=(I_down.min(), 0), xytext=(I_down.min(), f_down[order][0] - 3),
                arrowprops=dict(arrowstyle="->", color="#d1495b"))
    ax.text(I_up.min() - 0.1, 25, f"delayed\nonset\n{I_up.min():.1f}", fontsize=7, color="#2a9d8f", ha="right")
    ax.text(I_down.min() + 0.1, 25, f"fold of\ncycles\n{I_down.min():.2f}", fontsize=7, color="#d1495b")
    ax.axvline(hopf[0], color="0.4", ls=":", lw=1)
    ax.text(hopf[0] + 0.1, 92, f"Hopf {hopf[0]:.2f}", fontsize=8, color="0.3")
    ax.set_xlim(CFG.ramp_low, 14.0)
    ax.set_ylim(-8, 100)
    ax.set_xlabel(r"injected current I ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Onset is a jump, with hysteresis")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    I_dense = np.arange(0.0, 200.0, 0.5)
    ax.plot(I_dense, [equilibrium_max_real_eigenvalue(DEFAULT_PARAMS, I) for I in I_dense],
            color="#e9a03b", label=r"max Re $\lambda$ at equilibrium")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel(r"injected current I ($\mu$A/cm$^2$)")
    ax.set_ylabel(r"max Re $\lambda$ (1/ms)")
    ax2 = ax.twinx()
    ax2.plot(currents, step.amplitude, color=STATE_COLORS["V"], lw=1, label="oscillation amplitude")
    ax2.set_ylabel("late peak-to-peak V (mV)")
    ax2.grid(False)
    ax.set_title("Linear stability and amplitude")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.tight_layout()
    print("saved", save_figure(fig, "exp2_fi_curve"))


if __name__ == "__main__":
    apply_style()
    main()
