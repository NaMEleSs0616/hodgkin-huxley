"""Experiment 5: phase-plane analysis in the (V, n) plane.

Reduction (Rinzel 1985): m -> m_inf(V) because tau_m << tau_n, tau_h, and h -> h0 - k n
because h and n stay close to a line along trajectories (h0, k fitted on the full model's
limit cycle). The result is a planar ODE whose nullclines and orbits can be drawn exactly.

Figure:
    figures/exp5_phase_plane.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from hh import DEFAULT_PARAMS, HodgkinHuxley
from hh.config import PHASE_PLANE as CFG, STATE_COLORS
from hh.experiments import fit_h_n_relation, simulate
from hh.model import RinzelReduced, steady_state
from hh.plotting import apply_style, save_figure
from hh.solvers import rk4
from hh.stimuli import ConstantCurrent


def fixed_points(model: RinzelReduced) -> list[tuple[float, float, np.ndarray]]:
    """Intersections of the nullclines: dV(V, n_inf(V)) = 0, with Jacobian eigenvalues."""
    V = np.linspace(*CFG.V_range, 4001)
    g = model.dV(V, steady_state("n", V))
    out = []
    for i in np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:])):
        Vs = brentq(lambda v: float(model.dV(v, steady_state("n", v))), V[i], V[i + 1], xtol=1e-12)
        ns = float(steady_state("n", Vs))
        # 2x2 Jacobian by central differences
        e = 1e-6
        J = np.array([
            [(model.dV(Vs + e, ns) - model.dV(Vs - e, ns)) / (2 * e), (model.dV(Vs, ns + e) - model.dV(Vs, ns - e)) / (2 * e)],
            [(model.dn(Vs + e, ns) - model.dn(Vs - e, ns)) / (2 * e), (model.dn(Vs, ns + e) - model.dn(Vs, ns - e)) / (2 * e)],
        ], dtype=float)
        out.append((Vs, ns, np.linalg.eigvals(J)))
    return out


def panel(ax, I: float, h0: float, k: float) -> None:
    red = RinzelReduced(I_ext=I, h0=h0, k=k)
    V = np.linspace(*CFG.V_range, CFG.grid)
    n = np.linspace(*CFG.n_range, CFG.grid)
    VV, NN = np.meshgrid(V, n)
    dV, dn = red.dV(VV, NN), red.dn(VV, NN)

    ax.streamplot(VV, NN, dV, dn, color="0.82", density=1.1, linewidth=0.6, arrowsize=0.7)
    ax.contour(VV, NN, dV, levels=[0.0], colors=[STATE_COLORS["V"]], linewidths=2)
    ax.plot(V, steady_state("n", V), color=STATE_COLORS["n"], lw=2)
    ax.plot([], [], color=STATE_COLORS["V"], lw=2, label="V-nullcline (dV/dt = 0)")
    ax.plot([], [], color=STATE_COLORS["n"], lw=2, label=r"n-nullcline $n = n_\infty(V)$")

    for Vs, ns, lam in fixed_points(red):
        stable = np.all(lam.real < 0)
        kind = "focus" if np.any(np.abs(lam.imag) > 1e-9) else ("node" if stable or np.all(lam.real > 0) else "saddle")
        ax.plot(Vs, ns, "o", ms=8, mfc="k" if stable else "white", mec="k", zorder=5,
                label=f"{'stable' if stable else 'unstable'} {kind} ({Vs:.1f} mV)")
        print(f"  I = {I:g}: fixed point V = {Vs:.2f} mV, n = {ns:.4f}, eigenvalues {np.round(lam, 4)}")

    rest_full = HodgkinHuxley().resting_state()
    cmap = plt.get_cmap("plasma")
    if I == 0.0:
        for j, V0 in enumerate(CFG.kick_voltages):
            y0 = np.array([V0, rest_full[2]])
            sol = rk4(red.rhs, (0.0, CFG.t_end), y0, 0.01)
            ax.plot(sol.y[0], sol.y[1], color=cmap(0.15 + 0.7 * j / len(CFG.kick_voltages)), lw=1.3,
                    label=f"reduced, V(0) = {V0:g} mV")
            full = simulate(HodgkinHuxley(), (0.0, CFG.t_end),
                            np.array([V0, float(steady_state("m", V0)), rest_full[2], rest_full[3]]), dt=0.01)
            ax.plot(full.y[0], full.y[2], ":", color=cmap(0.15 + 0.7 * j / len(CFG.kick_voltages)), lw=1)
        ax.plot([], [], ":", color="0.3", label="full 4-D model (projection)")
        ax.set_title("I = 0: excitable (kicks above ~ -60 mV fire)")
    else:
        sol = rk4(red.rhs, (0.0, 200.0), np.array([-65.0, 0.35]), 0.01)
        late = sol.t > 100
        ax.plot(sol.y[0, late], sol.y[1, late], color="#d1495b", lw=2, label="reduced limit cycle")
        full = simulate(HodgkinHuxley(stimulus=ConstantCurrent(I)), (0.0, 200.0), dt=0.01)
        latef = full.t > 100
        ax.plot(full.y[0, latef], full.y[2, latef], ":", color="k", lw=1.2, label="full model limit cycle")
        per = lambda t, v: np.diff(t[np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0))]).mean()
        print(f"  I = {I:g}: period reduced {per(sol.t[late], sol.y[0, late]):.2f} ms vs full "
              f"{per(full.t[latef], full.y[0, latef]):.2f} ms")
        ax.set_title(rf"I = {I:g} $\mu$A/cm$^2$: unstable focus inside a limit cycle")
    ax.set_xlim(*CFG.V_range)
    ax.set_ylim(*CFG.n_range)
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("n")
    ax.legend(fontsize=6.5, loc="upper left")


def kick_threshold(peak_of) -> float:
    return brentq(lambda v: peak_of(v), -64.9, -40.0, xtol=1e-6)


def main() -> None:
    h0, k, rms = fit_h_n_relation(DEFAULT_PARAMS, I=CFG.fit_current)
    print(f"fitted on the full limit cycle: h = {h0:.4f} - {k:.4f} n (rms residual {rms:.4f}); "
          f"Rinzel's published values: 0.89 - 1.1 n")

    red = RinzelReduced(h0=h0, k=k)
    rest = HodgkinHuxley().resting_state()
    th_red = kick_threshold(lambda v: rk4(red.rhs, (0, 60), np.array([v, rest[2]]), 0.01).y[0].max())
    th_full = kick_threshold(lambda v: simulate(
        HodgkinHuxley(), (0, 60), np.array([v, float(steady_state("m", v)), rest[2], rest[3]]), dt=0.01).y[0].max())
    print(f"voltage-kick threshold (peak reaches 0 mV): reduced {th_red:.3f} mV, full {th_full:.3f} mV")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), width_ratios=[1.2, 1.2, 0.8])
    for ax, I in zip(axes[:2], CFG.currents):
        panel(ax, I, h0, k)

    ax = axes[2]
    full = simulate(HodgkinHuxley(stimulus=ConstantCurrent(CFG.fit_current)), (0.0, 200.0), dt=0.01)
    late = full.t > 100
    ax.plot(full.y[2, late], full.y[3, late], color=STATE_COLORS["h"], label="full model limit cycle")
    nn = np.linspace(0.3, 0.8, 10)
    ax.plot(nn, h0 - k * nn, "--", color="k", lw=1, label=f"fit h = {h0:.2f} - {k:.2f} n")
    rest = HodgkinHuxley().resting_state()
    ax.plot(rest[2], rest[3], "o", color="k", ms=5, label="rest")
    ax.set_xlabel("n")
    ax.set_ylabel("h")
    ax.set_title("Why h can be eliminated")
    ax.legend(fontsize=7)
    fig.tight_layout()
    print("saved", save_figure(fig, "exp5_phase_plane"))


if __name__ == "__main__":
    apply_style()
    main()
