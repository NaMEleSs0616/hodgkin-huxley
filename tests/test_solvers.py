"""Tests of the RK4 solver and the piecewise simulate() driver.

Convergence orders are measured, not assumed: if the error behaves like C*dt^p,
then halving dt divides it by 2^p, so p = log2(err(dt) / err(dt/2)).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import simpson, solve_ivp

from hh.config import DEFAULT_PARAMS
from hh.experiments import simulate, spike_times
from hh.model import HodgkinHuxley, ionic_currents
from hh.solvers import rk4
from hh.stimuli import ConstantCurrent, StepCurrent


def observed_orders(errors: list[float]) -> np.ndarray:
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


def tight_reference(model: HodgkinHuxley, t_span, y0, **kw):
    """High-accuracy scipy solution, integrated piece by piece across stimulus jumps."""
    edges = [t_span[0], *[b for b in getattr(model.stimulus, "breakpoints", ())
                          if t_span[0] < b < t_span[1]], t_span[1]]
    y = np.asarray(y0, dtype=float)
    for a, b in zip(edges[:-1], edges[1:]):
        piece = replace(model, stimulus=ConstantCurrent(model.stimulus(0.5 * (a + b))))
        sol = solve_ivp(piece.rhs, (a, b), y, method="DOP853", rtol=1e-13, atol=1e-13, **kw)
        y = sol.y[:, -1]
    return y


# ---------------------------------------------------------------------------
# RK4 on problems with exact solutions
# ---------------------------------------------------------------------------
def test_rk4_exact_for_cubic_quadrature():
    """For y' = f(t), RK4 reduces to Simpson's rule, which is exact for cubics."""
    f = lambda t, y: np.array([4 * t**3 - 3 * t**2 + 2 * t - 1])
    sol = rk4(f, (0.0, 2.0), [0.0], dt=0.5)
    exact = 2.0**4 - 2.0**3 + 2.0**2 - 2.0
    assert sol.y[0, -1] == pytest.approx(exact, rel=1e-14)


def test_rk4_fourth_order_on_exponential_decay():
    f = lambda t, y: -2.0 * y
    dts = [0.1, 0.05, 0.025, 0.0125]  # lambda*dt <= 0.2: inside the asymptotic regime
    errs = [abs(rk4(f, (0.0, 2.0), [1.0], dt).y[0, -1] - np.exp(-4.0)) for dt in dts]
    assert np.all(np.abs(observed_orders(errs) - 4.0) < 0.15)


def test_passive_membrane_matches_analytic_solution():
    """With g_Na = g_K = 0 the V equation is linear: C V' = I - g_L (V - E_L).

    Exact solution: V(t) = V_inf + (V0 - V_inf) exp(-t/tau), V_inf = E_L + I/g_L, tau = C/g_L.
    """
    p = replace(DEFAULT_PARAMS, g_Na=0.0, g_K=0.0)
    I = 3.0
    model = HodgkinHuxley(params=p, stimulus=ConstantCurrent(I))
    y0 = np.array([-65.0, 0.05, 0.3, 0.6])
    tau, V_inf = p.C_m / p.g_L, p.E_L + I / p.g_L
    T = 20.0
    exact = V_inf + (y0[0] - V_inf) * np.exp(-T / tau)

    dts = [0.4, 0.2, 0.1, 0.05]
    errs = [abs(rk4(model.rhs, (0.0, T), y0, dt).y[0, -1] - exact) for dt in dts]
    assert errs[-1] < 1e-9
    assert np.all(np.abs(observed_orders(errs) - 4.0) < 0.15)


# ---------------------------------------------------------------------------
# RK4 on the full HH system
# ---------------------------------------------------------------------------
def test_rk4_fourth_order_on_hh_spike():
    """Through a full action potential (smooth forcing), RK4 converges at order 4."""
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    y0 = model.resting_state()
    T = 15.0
    ref = tight_reference(model, (0.0, T), y0)
    dts = [0.04, 0.02, 0.01, 0.005]
    errs = [np.max(np.abs(rk4(model.rhs, (0.0, T), y0, dt).y[:, -1] - ref)) for dt in dts]
    orders = observed_orders(errs)
    assert np.all(orders > 3.7), orders
    assert orders[-1] == pytest.approx(4.0, abs=0.1)


def test_simulate_keeps_fourth_order_across_a_current_step():
    """Regression test: integrating straight through a current step gives order 1.

    simulate() restarts at the breakpoint, which must restore order 4.
    """
    model = HodgkinHuxley(stimulus=StepCurrent(10.0, t_on=1.0))
    y0 = model.resting_state()
    T = 15.0
    ref = tight_reference(model, (0.0, T), y0)
    dts = [0.04, 0.02, 0.01, 0.005]

    naive = [abs(rk4(model.rhs, (0.0, T), y0, dt).y[0, -1] - ref[0]) for dt in dts]
    piecewise = [abs(simulate(model, (0.0, T), y0, dt).y[0, -1] - ref[0]) for dt in dts]
    assert np.all(np.abs(observed_orders(naive) - 1.0) < 0.1)
    assert np.all(observed_orders(piecewise) > 3.7)


def test_rk4_spike_times_agree_with_scipy():
    model = HodgkinHuxley(stimulus=StepCurrent(10.0, t_on=5.0))
    T = 100.0
    ours = simulate(model, (0.0, T), dt=0.01)
    theirs = solve_ivp(model.rhs, (0.0, T), model.resting_state(), method="DOP853",
                       rtol=1e-11, atol=1e-11, t_eval=ours.t, max_step=0.1)
    s_ours = spike_times(ours.t, ours.y[0])
    s_theirs = spike_times(theirs.t, theirs.y[0])
    assert len(s_ours) == len(s_theirs) == 7
    np.testing.assert_allclose(s_ours, s_theirs, atol=1e-3)  # ms


def test_charge_conservation():
    """Integrating C dV/dt = I_ext - I_ion over time: C (V(T) - V(0)) = integral of net current.

    Checks that the solver's V trajectory is consistent with the currents it implies,
    through several spikes. Simpson's rule on the stored grid is 4th-order accurate.
    """
    I = 10.0
    model = HodgkinHuxley(stimulus=ConstantCurrent(I))
    sol = simulate(model, (0.0, 50.0), dt=0.005)
    I_net = I - np.sum(ionic_currents(sol.y), axis=0)
    charge_in = simpson(I_net, x=sol.t)  # nC/cm^2
    dV_times_C = DEFAULT_PARAMS.C_m * (sol.y[0, -1] - sol.y[0, 0])
    assert charge_in == pytest.approx(dV_times_C, abs=1e-4)


def test_gates_stay_in_unit_interval_under_strong_drive():
    model = HodgkinHuxley(stimulus=ConstantCurrent(np.array([-20.0, 5.0, 50.0, 200.0])))
    sol = simulate(model, (0.0, 50.0), dt=0.01, store_every=5)
    gates = sol.y[1:]
    assert gates.min() >= 0.0 and gates.max() <= 1.0


# ---------------------------------------------------------------------------
# Solver mechanics
# ---------------------------------------------------------------------------
def test_batch_integration_equals_individual_runs():
    amps = np.array([0.0, 3.0, 7.0, 15.0])
    batch = simulate(HodgkinHuxley(stimulus=StepCurrent(amps, t_on=2.0)), (0.0, 30.0))
    for k, a in enumerate(amps):
        single = simulate(HodgkinHuxley(stimulus=StepCurrent(a, t_on=2.0)), (0.0, 30.0))
        np.testing.assert_allclose(batch.y[:, k, :], single.y, rtol=1e-13, atol=1e-13)


def test_rk4_counts_function_evaluations():
    sol = rk4(lambda t, y: -y, (0.0, 1.0), [1.0], dt=0.1)
    assert sol.nfev == 40


def test_rk4_rejects_step_that_does_not_divide_interval():
    with pytest.raises(ValueError):
        rk4(lambda t, y: -y, (0.0, 1.0), [1.0], dt=0.3)


def test_store_every_keeps_endpoint():
    sol = rk4(lambda t, y: -y, (0.0, 1.0), [1.0], dt=0.1, store_every=3)
    np.testing.assert_allclose(sol.t, [0.0, 0.3, 0.6, 0.9, 1.0])
    assert sol.y.shape == (1, 5)


def test_simulate_output_grid():
    sol = simulate(HodgkinHuxley(stimulus=StepCurrent(5.0, t_on=1.234, t_off=7.0)), (0.0, 10.0), dt=0.01)
    assert sol.t[0] == 0.0 and sol.t[-1] == pytest.approx(10.0)
    assert np.all(np.diff(sol.t) > 0)
    assert np.any(np.isclose(sol.t, 1.234)) and np.any(np.isclose(sol.t, 7.0))
    assert np.max(np.diff(sol.t)) <= 0.01 + 1e-12
