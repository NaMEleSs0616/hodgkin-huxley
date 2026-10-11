"""Milestone 1: the model's ingredients and a basic spike train.

Figures:
    figures/gating_kinetics.png   x_inf(V) and tau_x(V) for m, n, h
    figures/spike_train_10uA.png  V, m, n, h under a 10 uA/cm^2 current step (RK4)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from hh import HodgkinHuxley, rk4
from hh.config import SOLVER, STATE_COLORS
from hh.experiments import spike_times
from hh.model import GATES, steady_state, time_constant
from hh.plotting import apply_style, plot_state_trace, save_figure
from hh.stimuli import StepCurrent


def gating_kinetics_figure() -> None:
    V = np.linspace(-100, 50, 1501)
    V_rest = HodgkinHuxley().resting_potential()
    fig, (ax_inf, ax_tau) = plt.subplots(1, 2, figsize=(9, 3.6))
    for g in GATES:
        ax_inf.plot(V, steady_state(g, V), color=STATE_COLORS[g], label=rf"${g}_\infty$")
        ax_tau.plot(V, time_constant(g, V), color=STATE_COLORS[g], label=rf"$\tau_{g}$")
    for ax in (ax_inf, ax_tau):
        ax.axvline(V_rest, color="0.5", ls=":", lw=1)
        ax.set_xlabel("V (mV)")
        ax.legend()
    ax_inf.set_ylabel("steady-state value")
    ax_tau.set_ylabel("time constant (ms)")
    ax_inf.set_title("Where each gate wants to be")
    ax_tau.set_title("How fast it gets there")
    ax_inf.text(V_rest + 2, 0.9, "rest", color="0.4")
    print("saved", save_figure(fig, "gating_kinetics"))


def spike_train_figure() -> None:
    stim = StepCurrent(10.0, t_on=10.0, t_off=90.0)
    model = HodgkinHuxley(stimulus=stim)
    sol = rk4(model.rhs, (0.0, 100.0), model.resting_state(), SOLVER.rk4_dt)
    I = np.array([stim(t) for t in sol.t])
    st = spike_times(sol.t, sol.y[0])
    print(f"{len(st)} spikes, ISIs = {np.round(np.diff(st), 2)} ms")
    fig = plot_state_trace(sol.t, sol.y, I, title=r"Repetitive firing at $I_{ext}$ = 10 $\mu$A/cm$^2$ (RK4, dt = 0.01 ms)")
    print("saved", save_figure(fig, "spike_train_10uA"))


if __name__ == "__main__":
    apply_style()
    gating_kinetics_figure()
    spike_train_figure()
