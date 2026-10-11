"""Experiment 1: current steps, thresholds, and the (quasi-)all-or-none response.

Figures:
    figures/exp1_step_responses.png  V, m, n, h for several step amplitudes
    figures/exp1_all_or_none.png     peak V vs pulse amplitude, and the graded response near threshold
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from hh import HodgkinHuxley
from hh.config import SPIKING as CFG, STATE_COLORS
from hh.experiments import pulse_threshold, simulate, step_threshold
from hh.model import GATES
from hh.plotting import apply_style, save_figure
from hh.stimuli import StepCurrent


def step_responses() -> None:
    amps = np.array(CFG.step_amplitudes)
    model = HodgkinHuxley(stimulus=StepCurrent(amps, t_on=CFG.step_t_on, t_off=CFG.step_t_off))
    sol = simulate(model, (0.0, CFG.step_t_end), dt=CFG.dt, store_every=4)

    fig, axes = plt.subplots(2, len(amps), figsize=(13, 5.2), sharex=True, sharey="row",
                             height_ratios=[3, 2])
    for j, a in enumerate(amps):
        ax_v, ax_g = axes[0, j], axes[1, j]
        ax_v.axvspan(CFG.step_t_on, CFG.step_t_off, color="0.93", lw=0)
        ax_v.plot(sol.t, sol.y[0, j], color=STATE_COLORS["V"])
        ax_v.set_title(rf"$I$ = {a:g} $\mu$A/cm$^2$")
        for i, g in enumerate(GATES, start=1):
            ax_g.plot(sol.t, sol.y[i, j], color=STATE_COLORS[g], label=g)
        ax_g.set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("V (mV)")
    axes[1, 0].set_ylabel("gating variable")
    axes[1, -1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Step currents (grey: current on): subthreshold, single spike, single spike, "
                 "repetitive, repetitive", y=1.0, fontsize=10)
    fig.tight_layout()
    print("saved", save_figure(fig, "exp1_step_responses"))


def all_or_none() -> None:
    th = float(pulse_threshold(width=CFG.pulse_width, hi=CFG.threshold_hi, window=CFG.pulse_window,
                               dt=CFG.dt, rel_tol=CFG.threshold_rel_tol))
    print(f"0.5 ms pulse threshold = {th:.10f} uA/cm^2 (charge {th * CFG.pulse_width:.3f} nC/cm^2)")

    def run(amps: np.ndarray, t_end: float = CFG.pulse_t_on + CFG.pulse_window):
        stim = StepCurrent(amps, t_on=CFG.pulse_t_on, t_off=CFG.pulse_t_on + CFG.pulse_width)
        return simulate(HodgkinHuxley(stimulus=stim), (0.0, t_end), dt=CFG.dt, store_every=2)

    lo, hi_, n = CFG.amplitude_scan
    scan = np.linspace(lo, hi_, n)
    sol = run(scan)
    peaks = sol.y[0].max(axis=1)

    deltas = np.array(CFG.log_deltas)
    near = run(np.concatenate([th * (1 + deltas), th * (1 - deltas)]))
    pk = near.y[0].max(axis=1)
    tpk = near.t[near.y[0].argmax(axis=1)] - CFG.pulse_t_on
    k = deltas.size
    above, below = slice(0, k), slice(k, 2 * k)
    slope_t = np.polyfit(np.log10(deltas[:-3]), tpk[above][:-3], 1)[0]
    rate = np.log(10) / -slope_t
    print(f"time-to-peak grows {-slope_t:.3f} ms per decade closer to threshold -> escape rate "
          f"lambda ~ {rate:.2f} /ms")
    for d, a, b in zip(deltas, pk[above], pk[below]):
        print(f"  delta = {d:.0e}: peak {a:7.2f} mV above threshold, {b:7.2f} mV below")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.plot(scan, peaks, ".", ms=3, color=STATE_COLORS["V"])
    ax.axvline(th, color="#d1495b", ls="--", lw=1)
    ax.text(th + 0.5, -20, rf"threshold {th:.3f} $\mu$A/cm$^2$", color="#d1495b", fontsize=9)
    ax.set_xlabel(rf"pulse amplitude ($\mu$A/cm$^2$, {CFG.pulse_width:g} ms)")
    ax.set_ylabel("peak V (mV)")
    ax.set_title("Looks all-or-none at normal resolution")

    ax = axes[1]
    ax.semilogx(deltas, pk[above], "o-", color="#2a9d8f", label="above threshold")
    ax.semilogx(deltas, pk[below], "s-", color="#6c757d", label="below threshold")
    ax.invert_xaxis()
    ax.set_xlabel(r"relative distance from threshold $|I/I_{th} - 1|$")
    ax.set_ylabel("peak V (mV)")
    ax.set_title("...but the peak is graded (FitzHugh's quasi-threshold)")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.semilogx(deltas, tpk[above], ":", color="#2a9d8f", lw=1)
    ax2.set_ylabel("time to peak (ms), dotted", color="0.4")
    ax2.grid(False)

    ax = axes[2]
    cmap = plt.get_cmap("coolwarm")
    show = [0, 4, 8]
    for i, idx in enumerate(show):
        for sign, sl, ls in ((+1, above, "-"), (-1, below, "--")):
            row = (np.arange(2 * k)[sl])[idx]
            col = cmap(0.85 if sign > 0 else 0.15)
            ax.plot(near.t - CFG.pulse_t_on, near.y[0, row], ls, color=col, alpha=1 - 0.25 * i, lw=1.2,
                    label=rf"$\delta = {'+' if sign > 0 else '-'}10^{{{int(np.log10(deltas[idx]))}}}$")
    ax.set_xlim(0, 20)
    ax.set_xlabel("time after pulse onset (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("Near threshold the decision is delayed")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    print("saved", save_figure(fig, "exp1_all_or_none"))


def thresholds() -> None:
    rheo = step_threshold(t_end=CFG.rheobase_t_end, hi=10.0, dt=CFG.dt * 2, rel_tol=1e-6)
    rep = step_threshold(t_end=CFG.repetitive_t_end, t_check=CFG.repetitive_t_check, lo=rheo, hi=15.0,
                         dt=CFG.dt * 2, rel_tol=1e-6)
    print(f"rheobase (any spike for a step from rest)        = {rheo:.4f} uA/cm^2")
    print(f"onset of sustained repetitive firing from rest   = {rep:.4f} uA/cm^2")


if __name__ == "__main__":
    apply_style()
    step_responses()
    all_or_none()
    thresholds()
