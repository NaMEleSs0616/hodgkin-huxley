"""Experiment 4: channel blockers as reduced maximal conductances.

TTX blocks voltage-gated Na channels  -> scale g_Na down.
TEA blocks voltage-gated K channels   -> scale g_K down.

A blocker that removes a fraction of the channels leaves the kinetics (alpha, beta) of the
remaining ones untouched, so it maps exactly onto a smaller g_bar.

Figure:
    figures/exp4_blockers.png
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from hh import DEFAULT_PARAMS as P, HodgkinHuxley
from hh.config import BLOCKERS as CFG, make_rng
from hh.experiments import (
    ap_metrics,
    equilibrium,
    equilibrium_max_real_eigenvalue,
    fi_curve,
    passive_ceiling,
    pulse_threshold,
    simulate,
)
from hh.plotting import apply_style, save_figure
from hh.stimuli import StepCurrent

TTX_CMAP, TEA_CMAP = plt.get_cmap("Blues_r"), plt.get_cmap("Oranges_r")


def condition(field: str, frac: float):
    return replace(P, **{field: getattr(P, field) * frac})


def ap_shapes(ax, field: str, fractions, cmap, label: str) -> None:
    print(f"{label}: fraction | V_rest | threshold | peak | amplitude | half-width | max dV/dt | AHP min")
    for k, f in enumerate(fractions):
        p = condition(field, f)
        y0 = equilibrium(p, 0.0)
        th = float(pulse_threshold(p, CFG.pulse_width, y0=y0, hi=CFG.threshold_hi,
                                   window=CFG.threshold_window, rel_tol=1e-6))
        sol = simulate(HodgkinHuxley(params=p, stimulus=StepCurrent(CFG.shape_multiple * th, 1.0,
                                                                     1.0 + CFG.pulse_width)),
                       (0.0, CFG.t_shape), y0, dt=0.005)
        mt = ap_metrics(sol.t, sol.y[0], y0[0])
        print(f"   {f:4.2f} | {y0[0]:7.2f} | {th:7.2f} | {mt['peak']:6.2f} | {mt['amplitude']:6.2f} | "
              f"{mt['half_width']:5.3f} | {mt['max_dvdt']:6.1f} | {mt['ahp_min']:7.2f}")
        ax.plot(sol.t, sol.y[0], color=cmap(0.75 * k / max(1, len(fractions) - 1)),
                label=f"{f:.0%} " + rf"$\bar g$, th {th:.0f}")
    ax.set_xlim(0, 25)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend(fontsize=7, loc="upper right", title=f"{label}: remaining", title_fontsize=7)


def fi_family(ax, field: str, fractions, cmap) -> None:
    I = np.arange(*CFG.fi_currents)
    fr = np.repeat(np.array(fractions), I.size)
    currents = np.tile(I, len(fractions))
    params = replace(P, **{field: getattr(P, field) * fr})  # one axon per trial (array parameters)
    y0 = np.column_stack([equilibrium(condition(field, f), 0.0) for f in fr])
    res = fi_curve(currents, params, y0, t_end=CFG.fi_t_end, t_transient=CFG.fi_t_transient, dt=CFG.dt)
    rate = res.spike_rate.reshape(len(fractions), I.size)
    for k, f in enumerate(fractions):
        onset = I[rate[k] > 0].min() if (rate[k] > 0).any() else np.nan
        print(f"   f-I {field} x{f:.2f}: repetitive firing from {onset:.1f} uA/cm^2, "
              f"rate at 20 uA/cm^2 = {rate[k, np.argmin(np.abs(I - 20))]:.0f} Hz")
        ax.plot(I, rate[k], color=cmap(0.75 * k / max(1, len(fractions) - 1)), label=f"{f:.0%}")
    ax.set_xlabel(r"injected current ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate (Hz)")
    ax.legend(fontsize=7, title="remaining", title_fontsize=7)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # ---------------- TTX
    ax = axes[0, 0]
    ap_shapes(ax, "g_Na", CFG.ttx_fractions, TTX_CMAP, "TTX")
    ax.set_title(r"TTX ($\bar g_{Na}$ ↓): smaller, slower upstroke")

    ax = axes[0, 1]
    ceiling = passive_ceiling(P, CFG.pulse_width, hi=1000.0, window=CFG.threshold_window, rel_tol=1e-6)
    scan = np.linspace(*CFG.ttx_scan)
    th = np.array([pulse_threshold(condition("g_Na", f), CFG.pulse_width, hi=CFG.threshold_hi,
                                   window=CFG.threshold_window, rel_tol=1e-4) for f in scan])
    blocked = th >= CFG.block_ratio * ceiling
    f_block = scan[~blocked].min()
    print(f"TTX: passive ceiling (no Na channels reach 0 mV) = {ceiling:.1f} uA/cm^2")
    print(f"TTX: regenerative APs lost below ~{f_block:.0%} of g_Na "
          f"(threshold >= {CFG.block_ratio:.0%} of the passive ceiling)")
    ax.plot(scan * 100, th, "o-", ms=3, color="#1f3a5f", label="pulse threshold")
    ax.axhline(ceiling, color="0.4", ls="--", lw=1, label="passive ceiling (no Na channels)")
    ax.axvspan(0, f_block * 100, color="#d1495b", alpha=0.12, lw=0, label="no regenerative AP")
    ax.set_xlabel(r"remaining $\bar g_{Na}$ (%)")
    ax.set_ylabel(r"threshold ($\mu$A/cm$^2$, 0.5 ms pulse)")
    ax.set_title("TTX raises threshold until APs fail")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    fi_family(ax, "g_Na", CFG.ttx_fractions, TTX_CMAP)
    ax.set_title("TTX: repetitive firing is lost first (none at <= 60%)")

    # ---------------- TEA
    ax = axes[1, 0]
    ap_shapes(ax, "g_K", CFG.tea_fractions, TEA_CMAP, "TEA")
    ax.set_title(r"TEA ($\bar g_K$ ↓): broader AP, depolarized rest")

    ax = axes[1, 1]
    scan = np.linspace(*CFG.tea_scan)
    eqs = np.column_stack([equilibrium(condition("g_K", f), 0.0) for f in scan])
    stable = np.array([equilibrium_max_real_eigenvalue(condition("g_K", f), 0.0) < 0 for f in scan])
    y0 = eqs + np.vstack([CFG.tea_kick_mV * make_rng().standard_normal(scan.size), np.zeros((3, scan.size))])
    params = replace(P, g_K=P.g_K * scan)
    sol = simulate(HodgkinHuxley(params=params), (0.0, CFG.tea_t_end), y0, dt=CFG.dt, store_every=10)
    late = sol.y[0][:, sol.t >= CFG.tea_t_transient]
    vmax, vmin = late.max(axis=1), late.min(axis=1)
    osc = vmax - vmin > 1.0
    print(f"TEA: rest unstable (spontaneous activity) for g_K fraction in "
          f"[{scan[~stable].min():.2f}, {scan[~stable].max():.2f}]; sustained oscillation for "
          f"[{scan[osc].min():.2f}, {scan[osc].max():.2f}]")
    for f in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
        k = np.argmin(np.abs(scan - f))
        print(f"   g_K x{scan[k]:.2f}: equilibrium V = {eqs[0, k]:7.2f} mV, "
              f"{'stable' if stable[k] else 'UNSTABLE'}, late V range [{vmin[k]:.1f}, {vmax[k]:.1f}]")
    ax.plot(scan[stable] * 100, eqs[0, stable], ".", color="k", ms=4, label="equilibrium (stable)")
    ax.plot(scan[~stable] * 100, eqs[0, ~stable], ".", color="0.6", ms=4, label="equilibrium (unstable)")
    ax.fill_between(scan * 100, np.where(osc, vmin, np.nan), np.where(osc, vmax, np.nan),
                    color="#e9a03b", alpha=0.35, lw=0, label="spontaneous oscillation (min-max V)")
    ax.set_xlabel(r"remaining $\bar g_K$ (%)")
    ax.set_ylabel("V (mV), no stimulus")
    ax.set_title("Strong TEA: spontaneous firing, then block")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1, 2]
    fi_family(ax, "g_K", CFG.tea_fractions, TEA_CMAP)
    ax.set_title("TEA: fires at lower currents, faster")

    fig.tight_layout()
    print("saved", save_figure(fig, "exp4_blockers"))


if __name__ == "__main__":
    apply_style()
    main()
