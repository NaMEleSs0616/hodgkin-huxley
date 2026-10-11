"""Milestone 3: validate the solvers and examine stiffness.

Figures:
    figures/convergence_order.png   error vs step size (Euler, midpoint, RK4) and error vs tolerance (DOPRI5)
    figures/work_precision.png      error vs function evaluations, spiking and quiescent regimes
    figures/step_size_failure.png   what goes wrong as the RK4 step grows
    figures/stiffness.png           Jacobian eigenvalues, stability-limited steps, explicit vs implicit cost
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from hh import HodgkinHuxley
from hh.config import SOLVER, STATE_COLORS, VALIDATION as CFG
from hh.experiments import membrane_derivative, simulate, spike_times
from hh.plotting import apply_style, save_figure
from hh.solvers import DOPRI5, EULER, MIDPOINT, RK4_TABLEAU, fixed_step_rk, rk4, stable_step_limit
from hh.stimuli import ConstantCurrent, StepCurrent

METHOD_STYLE = {
    "rk4": dict(color="#1f3a5f", marker="o", label="RK4 (ours, fixed step)"),
    "dopri45": dict(color="#d1495b", marker="s", label="DOPRI5 (ours, adaptive)"),
    "RK45": dict(color="#f4a6b0", marker="x", ms=8, label="scipy RK45"),
    "DOP853": dict(color="#8e6c8a", marker="^", label="scipy DOP853"),
    "LSODA": dict(color="#2a9d8f", marker="D", label="scipy LSODA"),
    "Radau": dict(color="#e9a03b", marker="v", label="scipy Radau (implicit)"),
    "BDF": dict(color="#6c757d", marker="P", label="scipy BDF (implicit)"),
}


def reference_state(model: HodgkinHuxley, t_end: float) -> np.ndarray:
    """High-accuracy final state, cross-checked between two unrelated methods."""
    kw = dict(rtol=SOLVER.reference_rtol, atol=SOLVER.reference_atol)
    a = simulate(model, (0, t_end), method=CFG.reference_method, **kw).y[:, -1]
    b = simulate(model, (0, t_end), method="Radau", **kw).y[:, -1]
    print(f"  reference check |DOP853 - Radau| = {np.max(np.abs(a - b)):.1e}")
    return a


# ---------------------------------------------------------------------------
def convergence_figure() -> None:
    print("convergence order")
    smooth = HodgkinHuxley(stimulus=ConstantCurrent(CFG.I_drive))
    y0 = smooth.resting_state()
    T = CFG.t_convergence
    ref = reference_state(smooth, T)
    floor = 2e-12  # |DOP853 - Radau| at 1e-12 tolerances, printed above
    # The same drive switched on at t = 1 ms, to show the cost of stepping through a jump.
    jump = HodgkinHuxley(stimulus=StepCurrent(CFG.I_drive, t_on=1.0))
    ref_jump = reference_state(jump, T)

    dts = np.array(CFG.convergence_dts)
    curves = {}
    for tab in (EULER, MIDPOINT, RK4_TABLEAU):
        curves[f"{tab.name} (order {tab.order})"] = [
            abs(fixed_step_rk(smooth.rhs, (0, T), y0, dt, tab).y[0, -1] - ref[0]) for dt in dts]
    curves["RK4 stepping through a current jump"] = [
        abs(rk4(jump.rhs, (0, T), y0, dt).y[0, -1] - ref_jump[0]) for dt in dts]
    for name, errs in curves.items():
        print(f"  {name:38s} observed orders {np.round(np.log2(np.array(errs[:-1]) / errs[1:]), 2)}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = ["#6c757d", "#2a9d8f", "#1f3a5f", "#d1495b"]
    for (name, errs), col in zip(curves.items(), colors):
        ls = "--" if "jump" in name else "-"
        ax.loglog(dts, errs, ls, marker="o", color=col, label=name)
    # reference slopes anchored mid-range (the smallest RK4 steps already sit on the floor)
    mid = 3
    for p, col in [(1, "#6c757d"), (2, "#2a9d8f"), (4, "#1f3a5f")]:
        anchor = curves[[k for k in curves if f"order {p}" in k][0]][mid]
        guide = anchor * (dts / dts[mid]) ** p
        ax.loglog(dts, guide * 3, ":", color=col, lw=1)
        ax.text(dts[0] * 1.08, guide[0] * 3, rf"$\propto \Delta t^{p}$", color=col, va="center", fontsize=9)
    ax.axhspan(1e-13, floor, color="0.85", alpha=0.6, lw=0)
    ax.text(dts[0] * 1.9, floor * 0.3, "reference / round-off floor", color="0.4", fontsize=8, ha="right")
    ax.set_xlabel(r"step size $\Delta t$ (ms)")
    ax.set_ylabel(r"$|V_{num}(T) - V_{ref}(T)|$ (mV)")
    ax.set_title(f"Fixed-step convergence through one spike (T = {T:g} ms)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
    ax.set_xlim(right=dts[0] * 2.2)
    ax.set_ylim(bottom=1e-13)

    tols = np.array(CFG.tolerances[:-1])
    for meth in ("dopri45", "RK45"):
        errs = [abs(simulate(smooth, (0, T), method=meth, rtol=tol, atol=tol).y[0, -1] - ref[0]) for tol in tols]
        st = METHOD_STYLE[meth]
        ax2.loglog(tols, errs, marker=st["marker"], color=st["color"], label=st["label"],
                   ls="-" if meth == "dopri45" else "none", ms=st.get("ms", 6))
    ax2.loglog(tols, tols * 10, ":", color="0.5", lw=1, label=r"error = 10 $\times$ tol")
    ax2.set_xlabel("tolerance (rtol = atol)")
    ax2.set_ylabel(r"$|V_{num}(T) - V_{ref}(T)|$ (mV)")
    ax2.set_title("Adaptive DOPRI5: error tracks the tolerance")
    ax2.legend(fontsize=8)
    print("  saved", save_figure(fig, "convergence_order"))


# ---------------------------------------------------------------------------
def work_precision_figure() -> None:
    print("work-precision")
    problems = {
        f"Spiking: {CFG.I_drive:g} $\\mu$A/cm$^2$ for {CFG.t_work_precision:g} ms":
            (HodgkinHuxley(stimulus=ConstantCurrent(CFG.I_drive)), CFG.t_work_precision, CFG.rk4_dts_spiking),
        f"Quiescent: {CFG.I_quiescent:g} $\\mu$A/cm$^2$ step, {CFG.t_quiescent:g} ms":
            (HodgkinHuxley(stimulus=StepCurrent(CFG.I_quiescent, t_on=CFG.t_on_quiescent)),
             CFG.t_quiescent, CFG.rk4_dts_quiescent),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, (model, T, rk4_dts)) in zip(axes, problems.items()):
        grid = np.arange(0.0, T + 1e-9, CFG.error_grid_spacing)
        V_ref = simulate(model, (0, T), method=CFG.reference_method, rtol=SOLVER.reference_rtol,
                         atol=SOLVER.reference_atol, t_eval=grid, max_step=CFG.reference_max_step).y[0]
        # independent check: RK4 with a tiny step, read off exactly on the grid
        chk = simulate(model, (0, T), dt=2.0**-10)
        V_chk = chk.y[0, np.isin(np.round(chk.t, 9), np.round(grid, 9))]
        ref_unc = np.max(np.abs(V_ref - V_chk))
        print(f"  reference check max |DOP853 - RK4(dt=2^-10)| on grid = {ref_unc:.1e}")
        for meth, st in METHOD_STYLE.items():
            work, err = [], []
            for s_ in (rk4_dts if meth == "rk4" else CFG.tolerances):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if meth == "rk4":
                        sol = simulate(model, (0, T), dt=s_)
                        on_grid = np.isin(np.round(sol.t, 9), np.round(grid, 9))
                        V = sol.y[0, on_grid]
                    else:
                        sol = simulate(model, (0, T), method=meth, rtol=s_, atol=s_, t_eval=grid)
                        V = sol.y[0]
                e = np.max(np.abs(V - V_ref))
                if np.isfinite(e) and V.size == grid.size:
                    work.append(sol.nfev)
                    err.append(max(e, 1e-11))
            ax.loglog(work, err, marker=st["marker"], color=st["color"], label=st["label"],
                      ms=st.get("ms", 5), ls="-" if meth != "RK45" else "none")
            print(f"  {title[:9]} {meth:8s} nfev {min(work):>7d}-{max(work):>8d}  err {min(err):.1e}-{max(err):.1e}")
        ax.axhspan(1e-12, max(ref_unc, 1e-12), color="0.85", alpha=0.7, lw=0)
        ax.text(0.02, 0.03, "below reference accuracy", transform=ax.transAxes, fontsize=8, color="0.4")
        ax.set_title(title)
        ax.set_xlabel("right-hand-side evaluations")
    axes[0].set_ylabel(r"max $|V - V_{ref}|$ on a 1 ms grid (mV)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Work-precision: lower-left is better", y=1.0)
    print("  saved", save_figure(fig, "work_precision"))


# ---------------------------------------------------------------------------
def step_size_failure_figure() -> None:
    print("step-size failure")
    model = HodgkinHuxley(stimulus=StepCurrent(CFG.I_drive, t_on=10.0))
    T = 100.0
    ref = simulate(model, (0, T), dt=0.001)  # every step stored: Hermite timing needs the fine grid
    st_ref = spike_times(ref.t, ref.y[0], dVdt=membrane_derivative(model, ref.t, ref.y))

    # stability limit along the reference trajectory, from the Jacobian eigenvalues
    sub = slice(None, None, 50)
    lams = np.linalg.eigvals(model.jacobian(0, ref.y[:, sub]))
    h_stab = np.array([stable_step_limit(RK4_TABLEAU, l) for l in lams])
    h_crit_linear = h_stab.min()

    scan = np.array(CFG.failure_scan_dts)
    timing_err, blew_up = [], []
    for dt in scan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = simulate(model, (0, T), dt=dt)
        ok = np.isfinite(s.y).all(axis=0)
        blew_up.append(not ok.all())
        st = spike_times(s.t, s.y[0], dVdt=membrane_derivative(model, s.t, s.y)) if ok.all() else []
        k = min(len(st), len(st_ref))
        timing_err.append(np.max(np.abs(st[:k] - st_ref[:k])) if ok.all() else np.nan)
    blew_up = np.array(blew_up)
    dt_blow = scan[blew_up].min()
    print(f"  linear stability limit min_t h_stab = {h_crit_linear:.4f} ms; first blow-up in scan at dt = {dt_blow}")

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax_tr = fig.add_subplot(gs[0, :])
    cmap = plt.get_cmap("viridis")
    for i, dt in enumerate(CFG.failure_dts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = simulate(model, (0, 40.0), dt=dt)
        V = s.y[0]
        ok = np.isfinite(V) & (np.abs(V) < 200)
        col = cmap(i / (len(CFG.failure_dts) - 1) * 0.9)
        label = f"dt = {dt:g} ms"
        if not ok.all():
            j = np.argmin(ok)
            label += f" (blows up at t = {s.t[j]:.1f} ms)"
            ax_tr.plot(s.t[j - 1], np.clip(V[j - 1], -100, 60), "X", color=col, ms=10)
        ax_tr.plot(s.t[ok], V[ok], color=col, label=label, lw=1.2)
    ax_tr.set_xlim(8, 40)
    ax_tr.set_ylim(-90, 60)
    ax_tr.set_xlabel("time (ms)")
    ax_tr.set_ylabel("V (mV)")
    ax_tr.set_title(r"RK4 at growing step sizes (10 $\mu$A/cm$^2$ from t = 10 ms)")
    ax_tr.legend(fontsize=8, loc="upper right", ncols=2, framealpha=0.9)

    ax_err = fig.add_subplot(gs[1, 0])
    good = ~blew_up
    ax_err.loglog(scan[good], np.array(timing_err)[good], "o-", color=STATE_COLORS["V"], ms=4)
    ax_err.axvspan(dt_blow, scan.max() * 1.1, color="#d1495b", alpha=0.15)
    ax_err.axvline(h_crit_linear, color="#d1495b", ls="--", lw=1)
    ax_err.text(h_crit_linear * 0.95, 0.5, "linear stability\nlimit at spike peak", ha="right",
                va="top", fontsize=8, color="#d1495b")
    ax_err.text(dt_blow * 1.02, 1e-6, "blows\nup", fontsize=8, color="#d1495b")
    ax_err.set_xlabel(r"step size $\Delta t$ (ms)")
    ax_err.set_ylabel("max spike-time error (ms)")
    ax_err.set_title("Accuracy is excellent until stability is lost")

    ax_st = fig.add_subplot(gs[1, 1])
    ax_st.semilogy(ref.t[sub], h_stab, color="#d1495b", label=r"RK4 stable step $h_{stab}(t)$")
    ax_st.axhline(0.01, color="0.4", ls=":", lw=1, label="default dt = 0.01 ms")
    ax_st.set_xlabel("time (ms)")
    ax_st.set_ylabel(r"$h_{stab}$ (ms)")
    ax_st.set_xlim(8, 40)
    ax_v = ax_st.twinx()
    ax_v.plot(ref.t, ref.y[0], color=STATE_COLORS["V"], alpha=0.25, lw=1)
    ax_v.set_ylabel("V (mV)", color="0.5")
    ax_v.grid(False)
    ax_st.set_title("Stability limit collapses at each spike peak")
    ax_st.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    print("  saved", save_figure(fig, "step_size_failure"))


# ---------------------------------------------------------------------------
def stiffness_figure() -> None:
    print("stiffness")
    model = HodgkinHuxley(stimulus=ConstantCurrent(CFG.I_drive))
    T = CFG.stiffness_t_end
    ref = simulate(model, (0, T), dt=0.002, store_every=5)
    lams = np.linalg.eigvals(model.jacobian(0, ref.y))
    re = np.sort(lams.real, axis=1)  # ascending: most negative first
    lam_rest = np.linalg.eigvals(model.jacobian(0, HodgkinHuxley().resting_state()))
    print(f"  most negative Re(lambda) on orbit = {re[:, 0].min():.1f} /ms; stiffness ratio at rest = "
          f"{np.abs(lam_rest.real).max() / np.abs(lam_rest.real).min():.1f}; fraction of orbit with a "
          f"positive eigenvalue = {(re > 0).any(axis=1).mean():.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    ax = axes[0, 0]
    ax.plot(ref.t, ref.y[0], color=STATE_COLORS["V"])
    ax.set_ylabel("V (mV)")
    ax.set_title(rf"Trajectory ({CFG.I_drive:g} $\mu$A/cm$^2$)")
    ax.set_xlabel("time (ms)")

    ax = axes[0, 1]
    for k in range(4):
        ax.semilogy(ref.t, np.abs(re[:, k]), lw=1, label=rf"$|\mathrm{{Re}}\,\lambda_{k + 1}|$")
    pos = (re > 0).any(axis=1)
    ax.fill_between(ref.t, 1e-2, 1e2, where=pos, color="#e9a03b", alpha=0.15, label=r"some Re $\lambda > 0$")
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$|\mathrm{Re}\,\lambda|$ (1/ms)")
    ax.set_title("Jacobian eigenvalues along the orbit")
    ax.legend(fontsize=7, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    ax = axes[1, 0]
    for tol, col in zip(CFG.stiffness_tols, ("#d1495b", "#1f3a5f")):
        s = simulate(model, (0, T), method="dopri45", rtol=tol, atol=tol)
        h = np.diff(s.t)
        lam_steps = np.linalg.eigvals(model.jacobian(0, s.y[:, :-1]))
        h_st = np.array([stable_step_limit(DOPRI5, l) for l in lam_steps])
        frac = np.mean(h > 0.7 * h_st)
        ax.semilogy(s.t[:-1], h, ".", ms=3, color=col,
                    label=f"DOPRI5 steps, tol={tol:.0e} ({frac:.0%} near stability limit)")
        print(f"  tol {tol:.0e}: {frac:.0%} of steps exceed 0.7 h_stab")
    lam_ref = np.linalg.eigvals(model.jacobian(0, ref.y[:, ::4]))
    ax.semilogy(ref.t[::4], [stable_step_limit(DOPRI5, l) for l in lam_ref], color="k", lw=1,
                label="DOPRI5 stability limit")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("step size (ms)")
    ax.set_title("Loose tolerance: stability, not accuracy, sets the step")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)

    ax = axes[1, 1]
    quiet = HodgkinHuxley(stimulus=StepCurrent(CFG.I_quiescent, t_on=CFG.t_on_quiescent))
    names = ["dopri45", "RK45", "DOP853", "LSODA", "Radau", "BDF"]
    x = np.arange(len(names))
    for j, (tol, hatch) in enumerate(zip((1e-3, 1e-6), ("", "//"))):
        counts = [simulate(quiet, (0, CFG.t_quiescent), method=n, rtol=tol, atol=tol).nfev for n in names]
        ax.bar(x + (j - 0.5) * 0.38, counts, 0.38, color=[METHOD_STYLE[n]["color"] for n in names],
               hatch=hatch, edgecolor="white", label=f"tol = {tol:.0e}")
        print(f"  quiescent tol {tol:.0e}: " + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    ax.set_yscale("log")
    ax.set_xticks(x, ["DOPRI5\n(ours)", "RK45", "DOP853", "LSODA", "Radau", "BDF"], fontsize=8)
    ax.set_ylabel("RHS evaluations")
    ax.set_title(f"{CFG.t_quiescent:g} ms without spikes: implicit wins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    print("  saved", save_figure(fig, "stiffness"))


if __name__ == "__main__":
    apply_style()
    convergence_figure()
    work_precision_figure()
    step_size_failure_figure()
    stiffness_figure()
