"""Experiment 6 (extension): temperature via a Q10 factor on every rate constant.

alpha, beta -> phi * alpha, phi * beta with phi = Q10^((T - 6.3 degC)/10), Q10 = 3.
Steady states x_inf are unchanged (phi cancels); every gating time constant shrinks by phi.
The membrane time constant C/g does not scale, which is what eventually breaks excitability.

Figure:
    figures/exp6_temperature.png
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from hh import DEFAULT_PARAMS as P, HodgkinHuxley
from hh.config import TEMPERATURE as CFG
from hh.experiments import (
    ap_metrics,
    equilibrium,
    equilibrium_max_real_eigenvalue,
    fi_curve,
    hopf_interval,
    pulse_threshold,
    simulate,
)
from hh.plotting import apply_style, save_figure
from hh.stimuli import StepCurrent

ADAPTIVE = dict(method="dopri45", rtol=1e-8, atol=1e-8)


def at(T: float):
    return replace(P, temperature=T, q10=CFG.q10)


def max_instability(T: float) -> float:
    """max over I of the largest real eigenvalue at the equilibrium (> 0: some current destabilizes rest)."""
    p = at(T)
    grid = np.arange(0.0, 200.0, 1.0)
    vals = np.array([equilibrium_max_real_eigenvalue(p, I) for I in grid])
    i = int(np.argmax(vals))
    res = minimize_scalar(lambda I: -equilibrium_max_real_eigenvalue(p, I),
                          bounds=(grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]), method="bounded",
                          options=dict(xatol=1e-6))
    return -res.fun


def main() -> None:
    temps = np.array(CFG.temperatures)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    cmap = plt.get_cmap("coolwarm")

    ax = axes[0, 0]
    metrics = []
    print("T (degC) | phi | threshold | peak | half-width (ms) | AHP min")
    for k, T in enumerate(temps):
        p = at(T)
        y0 = equilibrium(p, 0.0)
        th = float(pulse_threshold(p, CFG.pulse_width, y0=y0, hi=CFG.threshold_hi, window=30.0,
                                   rel_tol=1e-6, **ADAPTIVE))
        sol = simulate(HodgkinHuxley(params=p, stimulus=StepCurrent(1.5 * th, 1.0, 1.0 + CFG.pulse_width)),
                       (0.0, 30.0), y0, t_eval=np.arange(0.0, 30.0, 0.002), **ADAPTIVE)
        mt = ap_metrics(sol.t, sol.y[0], y0[0])
        metrics.append((th, mt["peak"], mt["half_width"], mt["ahp_min"]))
        print(f"   {T:5.1f} | {p.phi:5.2f} | {th:7.2f} | {mt['peak']:6.2f} | {mt['half_width']:6.3f} | "
              f"{mt['ahp_min']:7.2f}")
        ax.plot(sol.t - 1.0, sol.y[0], color=cmap(k / (len(temps) - 1)), label=f"{T:g} °C")
    ax.set_xlim(-0.5, 8)
    ax.set_xlabel("time after pulse onset (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("Warmer: faster, briefer, smaller action potentials")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    th, peak, hw, ahp = np.array(metrics).T
    ax.semilogy(temps, hw, "o-", color="#1f3a5f", label="half-width (ms)")
    ax.semilogy(temps, th, "s-", color="#d1495b", label=r"threshold ($\mu$A/cm$^2$)")
    ax.semilogy(temps, hw[0] / np.array([at(T).phi for T in temps]), ":", color="#1f3a5f", lw=1,
                label=r"half-width if it scaled like $1/\phi$")
    ax.set_xlabel("temperature (°C)")
    ax.set_title("Width shrinks more slowly than the kinetics speed up")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    T_grid = np.arange(6.3, 30.01, 0.25)
    lo, hi_ = [], []
    for T in T_grid:
        h = hopf_interval(at(T), np.arange(0.0, 200.0, 0.5))
        lo.append(h[0] if h else np.nan)
        hi_.append(h[1] if h else np.nan)
    lo, hi_ = np.array(lo), np.array(hi_)
    T_star = brentq(max_instability, 25.0, 30.0, xtol=1e-6)
    print(f"Hopf interval at 6.3 degC: ({lo[0]:.2f}, {hi_[0]:.2f}); it shrinks with temperature and "
          f"closes at T* = {T_star:.3f} degC: above it no constant current can destabilize rest")
    ax.fill_betweenx(T_grid, lo, hi_, color="#e9a03b", alpha=0.35, lw=0, label="rest unstable (between the Hopf points)")
    ax.plot(lo, T_grid, color="#e9a03b")
    ax.plot(hi_, T_grid, color="#e9a03b")
    ax.axhline(T_star, color="#d1495b", ls="--", lw=1)
    ax.text(100, T_star + 0.4, f"heat block of repetitive firing, T* = {T_star:.2f} °C", ha="center",
            fontsize=8, color="#d1495b")
    ax.set_xlabel(r"injected current ($\mu$A/cm$^2$)")
    ax.set_ylabel("temperature (°C)")
    ax.set_title("Hopf bifurcations merge as temperature rises")
    ax.set_ylim(6.3, 32)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 1]
    I = np.arange(*CFG.fi_currents)
    for T in CFG.fi_temperatures:
        p = at(T)
        dt = CFG.base_dt / max(1.0, p.phi)
        res = fi_curve(I, p, t_end=300.0, t_transient=100.0, dt=dt, store_every=max(1, int(0.05 / dt)))
        onset = I[res.spike_rate > 0].min() if (res.spike_rate > 0).any() else np.nan
        print(f"f-I at {T:g} degC (dt = {dt:.5f} ms): spiking from {onset:g} uA/cm^2, "
              f"max {res.spike_rate.max():.0f} Hz; oscillation max {res.oscillation_freq.max():.0f} Hz")
        col = cmap((T - temps.min()) / (temps.max() - temps.min()))
        ax.plot(I, res.spike_rate, color=col, label=f"{T:g} °C, spikes")
        ax.plot(I, res.oscillation_freq, "--", color=col, lw=1)
    ax.plot([], [], "--", color="0.4", lw=1, label="oscillation frequency")
    ax.set_xlabel(r"injected current ($\mu$A/cm$^2$)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Warm: faster oscillations; at 25 °C too small to reach 0 mV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    print("saved", save_figure(fig, "exp6_temperature"))


if __name__ == "__main__":
    apply_style()
    main()
