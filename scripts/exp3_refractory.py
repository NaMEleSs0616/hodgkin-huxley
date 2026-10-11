"""Experiment 3: absolute and relative refractory periods from a two-pulse protocol.

A conditioning pulse (2x resting threshold) fires a spike; a second 0.5 ms pulse follows
after a variable interval, and we search for the smallest second-pulse amplitude that
fires again. "Absolute" = no spike for any amplitude up to the cap; "relative" = threshold
elevated by more than 5 %.

Validity check: the cap is chosen so that a membrane with no Na channels cannot reach
0 mV from any of the starting states, so every detected second spike is regenerative.

Figure:
    figures/exp3_refractory.png
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from hh import DEFAULT_PARAMS, HodgkinHuxley
from hh.config import REFRACTORY as CFG, STATE_COLORS
from hh.experiments import fires, refractory_curve, simulate
from hh.plotting import apply_style, save_figure
from hh.stimuli import PulsePair, StepCurrent


def main() -> None:
    intervals = np.round(np.arange(*CFG.intervals), 6)
    res = refractory_curve(width=CFG.pulse_width, conditioning_multiple=CFG.conditioning_multiple,
                           t_first=CFG.t_first, intervals=intervals, cap_multiple=CFG.cap_multiple,
                           recovered_ratio=CFG.recovered_ratio, window_after=CFG.window_after,
                           dt=CFG.dt, rel_tol=CFG.rel_tol)
    cap = CFG.cap_multiple * res.resting_threshold

    # control: the cap pulse into a Na-free membrane, from every starting state
    no_na = HodgkinHuxley(params=replace(DEFAULT_PARAMS, g_Na=0.0),
                          stimulus=StepCurrent(np.full(res.intervals.size, cap), 0.0, CFG.pulse_width))
    ctrl = simulate(no_na, (0.0, CFG.window_after), res.states, dt=CFG.dt, store_every=2)
    n_false = int(fires(ctrl.y[0]).sum())
    print(f"resting threshold {res.resting_threshold:.4f} uA/cm^2; cap {cap:.1f} uA/cm^2")
    print(f"control (g_Na = 0, cap amplitude): {n_false} of {res.intervals.size} starts cross 0 mV "
          f"-> {'criterion valid' if n_false == 0 else 'CRITERION INVALID'}")
    assert n_false == 0

    ratio = res.thresholds / res.resting_threshold
    finite = np.isfinite(ratio)
    i_min = np.argmin(np.where(finite, ratio, np.inf))
    print(f"absolute refractory period: up to {res.absolute_end:.1f} ms after the conditioning pulse")
    print(f"relative refractory period: until {res.relative_end:.1f} ms (threshold back within "
          f"{100 * (CFG.recovered_ratio - 1):.0f} %)")
    print(f"supernormal phase: threshold falls to {ratio[i_min]:.3f}x rest at {res.intervals[i_min]:.1f} ms")
    for iv in (5.0, 7.0, 10.0, 15.0, 20.0, 30.0):
        k = np.argmin(np.abs(res.intervals - iv))
        V, m, n, h = res.states[:, k]
        print(f"  {res.intervals[k]:5.1f} ms: ratio {ratio[k]:6.3f}  V {V:7.2f}  h {h:.3f}  n {n:.3f}")

    rest = HodgkinHuxley().resting_state()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    ex = np.array(CFG.example_intervals)
    amp = CFG.example_multiple * res.resting_threshold
    # for display both pulses share one suprathreshold amplitude
    sol = simulate(HodgkinHuxley(stimulus=PulsePair(amp, CFG.pulse_width, CFG.t_first, ex)),
                   (0.0, CFG.t_first + ex.max() + 15.0), dt=CFG.dt, store_every=4)
    cmap = plt.get_cmap("viridis")
    for k, iv in enumerate(ex):
        ax.plot(sol.t, sol.y[0, k], color=cmap(k / len(ex) * 0.9), lw=1.1, label=f"{iv:g} ms")
        ax.plot(CFG.t_first + iv, -88, "^", color=cmap(k / len(ex) * 0.9), ms=6)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title(rf"Two pulses of {amp:.0f} $\mu$A/cm$^2$ ($\blacktriangle$ = 2nd)")
    ax.legend(title="interval", fontsize=7, title_fontsize=8, loc="upper right")
    ax.set_ylim(-92, 50)

    ax = axes[1]
    ax.axvspan(res.intervals.min(), res.absolute_end, color="#d1495b", alpha=0.25, lw=0, label="absolute")
    ax.axvspan(res.absolute_end, res.relative_end, color="#e9a03b", alpha=0.2, lw=0, label="relative")
    ax.semilogy(res.intervals[finite], ratio[finite], color=STATE_COLORS["V"])
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.axhline(CFG.cap_multiple, color="#d1495b", ls=":", lw=1)
    ax.text(res.absolute_end + 1.0, CFG.cap_multiple * 1.07, "search cap", fontsize=8, color="#d1495b")
    ax.annotate("supernormal", xy=(res.intervals[i_min], ratio[i_min]),
                xytext=(res.intervals[i_min] + 4, 0.65), fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.set_ylim(0.6, 13)
    ax.set_xlabel("interval between pulse onsets (ms)")
    ax.set_ylabel("2nd-pulse threshold / resting threshold")
    ax.set_title("Threshold recovery")
    ax.legend(fontsize=8, loc="center right", title="refractory", title_fontsize=8)

    ax = axes[2]
    _, _, n, h = res.states
    ax.plot(res.intervals, h, color=STATE_COLORS["h"], label="h (Na inactivation gate)")
    ax.plot(res.intervals, n, color=STATE_COLORS["n"], label="n (K activation gate)")
    ax.axhline(rest[3], color=STATE_COLORS["h"], ls=":", lw=1)
    ax.axhline(rest[2], color=STATE_COLORS["n"], ls=":", lw=1)
    ax.plot(res.intervals, n**4, color=STATE_COLORS["n"], ls="--", lw=1,
            label=r"$n^4$ (K conductance / $\bar g_K$)")
    ax.axvspan(res.intervals.min(), res.absolute_end, color="#d1495b", alpha=0.12, lw=0)
    ax.axvspan(res.absolute_end, res.relative_end, color="#e9a03b", alpha=0.1, lw=0)
    ax.set_xlabel("interval between pulse onsets (ms)")
    ax.set_ylabel("gate value at the 2nd pulse")
    ax.set_title("Why: h recovers slowly, n relaxes slowly")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    print("saved", save_figure(fig, "exp3_refractory"))


if __name__ == "__main__":
    apply_style()
    main()
