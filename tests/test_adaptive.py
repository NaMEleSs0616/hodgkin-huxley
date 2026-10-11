"""Tests of the Butcher tableaux, stability analysis, the adaptive Dormand-Prince
solver, the analytic Jacobian, and agreement with scipy."""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from hh.experiments import METHODS, bisect_threshold, fires, simulate, spike_times
from hh.model import HodgkinHuxley
from hh.solvers import (
    DOPRI5,
    EULER,
    MIDPOINT,
    RK4_TABLEAU,
    _DP_B4,
    _DP_P,
    dopri45,
    fixed_step_rk,
    rk4,
    stability_polynomial,
    stable_step_limit,
)
from hh.stimuli import ConstantCurrent, StepCurrent


# ---------------------------------------------------------------------------
# Order conditions (Butcher's rooted-tree conditions up to order 5)
# ---------------------------------------------------------------------------
def order_conditions(A, b, c):
    """Return [(achieved, required, order)] for every rooted tree up to order 5."""
    Ac = A @ c
    return [
        (b.sum(), 1, 1),
        (b @ c, 1 / 2, 2),
        (b @ c**2, 1 / 3, 3), (b @ Ac, 1 / 6, 3),
        (b @ c**3, 1 / 4, 4), (b @ (c * Ac), 1 / 8, 4), (b @ (A @ c**2), 1 / 12, 4),
        (b @ (A @ Ac), 1 / 24, 4),
        (b @ c**4, 1 / 5, 5), (b @ (c**2 * Ac), 1 / 10, 5), (b @ (c * (A @ c**2)), 1 / 15, 5),
        (b @ (c * (A @ Ac)), 1 / 30, 5), (b @ Ac**2, 1 / 20, 5), (b @ (A @ c**3), 1 / 20, 5),
        (b @ (A @ (c * Ac)), 1 / 40, 5), (b @ (A @ A @ c**2), 1 / 60, 5),
        (b @ (A @ A @ Ac), 1 / 120, 5),
    ]


@pytest.mark.parametrize("tableau", [EULER, MIDPOINT, RK4_TABLEAU, DOPRI5], ids=lambda t: t.name)
def test_tableau_satisfies_order_conditions_up_to_its_order(tableau):
    np.testing.assert_allclose(tableau.A.sum(axis=1), tableau.c, atol=1e-15)  # consistency
    for got, want, order in order_conditions(tableau.A, tableau.b, tableau.c):
        if order <= tableau.order:
            assert got == pytest.approx(want, rel=1e-13), f"order-{order} condition"


def test_dormand_prince_embedded_pair_is_order_4_not_5():
    conds = order_conditions(DOPRI5.A, _DP_B4, DOPRI5.c)
    assert all(abs(g - w) < 1e-13 for g, w, o in conds if o <= 4)
    assert any(abs(g - w) > 1e-6 for g, w, o in conds if o == 5)


def test_dense_output_reproduces_step_endpoint_and_derivative():
    np.testing.assert_allclose(_DP_P.sum(axis=1), DOPRI5.b, atol=1e-14)  # x = 1 gives y_new
    deriv_at_1 = _DP_P @ np.arange(1, 5)
    np.testing.assert_allclose(deriv_at_1, np.eye(7)[6], atol=1e-13)  # slope = k7 = f(y_new)


# ---------------------------------------------------------------------------
# Linear stability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tableau", [EULER, MIDPOINT, RK4_TABLEAU, DOPRI5], ids=lambda t: t.name)
def test_stability_polynomial_matches_exp_taylor_series_to_order(tableau):
    coeffs = stability_polynomial(tableau)
    for k in range(tableau.order + 1):
        assert coeffs[k] == pytest.approx(1 / factorial(k), rel=1e-13)


def test_dopri5_stability_polynomial_extra_term():
    assert stability_polynomial(DOPRI5)[6] == pytest.approx(1 / 600, rel=1e-12)


@pytest.mark.parametrize(
    "tableau, limit", [(EULER, 2.0), (MIDPOINT, 2.0), (RK4_TABLEAU, 2.785293563), (DOPRI5, 3.306567)]
)
def test_real_axis_stability_interval(tableau, limit):
    assert stable_step_limit(tableau, [-1.0]) == pytest.approx(limit, rel=1e-6)


def test_rk4_blows_up_just_beyond_its_stability_limit():
    lam = -10.0
    h_crit = stable_step_limit(RK4_TABLEAU, [lam])
    f = lambda t, y: lam * y
    for h, should_decay in [(0.95 * h_crit, True), (1.05 * h_crit, False)]:
        n = 400
        y_end = rk4(f, (0.0, n * h), [1.0], h).y[0, -1]
        assert (abs(y_end) < 1.0) == should_decay


# ---------------------------------------------------------------------------
# Fixed-step generic RK
# ---------------------------------------------------------------------------
def test_generic_rk_with_rk4_tableau_equals_handwritten_rk4():
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    y0 = model.resting_state()
    a = rk4(model.rhs, (0.0, 5.0), y0, 0.01).y
    b = fixed_step_rk(model.rhs, (0.0, 5.0), y0, 0.01, RK4_TABLEAU).y
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tableau", [EULER, MIDPOINT], ids=lambda t: t.name)
def test_low_order_methods_converge_at_their_order(tableau):
    f = lambda t, y: -2.0 * y + np.sin(t)
    exact = lambda t: (np.exp(-2 * t) * 1.2) + (2 * np.sin(t) - np.cos(t)) / 5
    # T = 1, not 2: Euler's leading error coefficient changes sign near t = 2, where the
    # h^2 term would dominate and the measured "order" would be meaningless.
    errs = [abs(fixed_step_rk(f, (0.0, 1.0), [exact(0.0)], dt, tableau).y[0, -1] - exact(1.0))
            for dt in (0.01, 0.005, 0.0025)]
    orders = np.log2(np.array(errs[:-1]) / errs[1:])
    np.testing.assert_allclose(orders, tableau.order, atol=0.05)


# ---------------------------------------------------------------------------
# Adaptive Dormand-Prince
# ---------------------------------------------------------------------------
def test_dopri_exact_for_quartic_quadrature():
    """b5 integrates polynomials of degree <= 4 exactly, so one step suffices."""
    f = lambda t, y: np.array([5 * t**4 - 3 * t**2 + 1.0])
    sol = dopri45(f, (0.0, 1.5), [0.0], rtol=1e-12, atol=1e-12, h0=1.5)
    assert sol.y[0, -1] == pytest.approx(1.5**5 - 1.5**3 + 1.5, rel=1e-14)


def test_dopri_matches_scipy_rk45_step_for_step():
    """Same method, same I-controller, same starting step: identical cost, same answer."""
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    y0 = model.resting_state()
    for tol in (1e-4, 1e-7):
        ours = dopri45(model.rhs, (0.0, 40.0), y0, rtol=tol, atol=tol)
        theirs = solve_ivp(model.rhs, (0.0, 40.0), y0, method="RK45", rtol=tol, atol=tol)
        assert ours.nfev == theirs.nfev
        np.testing.assert_allclose(ours.t, theirs.t, rtol=1e-9)
        np.testing.assert_allclose(ours.y, theirs.y, atol=1e-6)


def test_dopri_error_is_controlled_by_tolerance():
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    y0 = model.resting_state()
    ref = solve_ivp(model.rhs, (0, 30), y0, method="DOP853", rtol=1e-13, atol=1e-13).y[0, -1]
    errs = {tol: abs(dopri45(model.rhs, (0, 30), y0, rtol=tol, atol=tol).y[0, -1] - ref)
            for tol in (1e-5, 1e-7, 1e-9)}
    assert errs[1e-5] > errs[1e-7] > errs[1e-9]
    assert errs[1e-9] < 1e-6
    # 100x tighter tolerance -> roughly 10-1000x smaller error
    assert 10 < errs[1e-5] / errs[1e-7] < 1e4


@pytest.mark.parametrize("controller", ["I", "PI"])
def test_dense_output_accurate_between_steps(controller):
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    y0 = model.resting_state()
    t_eval = np.linspace(0, 30, 3001)
    ours = dopri45(model.rhs, (0, 30), y0, rtol=1e-9, atol=1e-9, t_eval=t_eval, controller=controller)
    ref = solve_ivp(model.rhs, (0, 30), y0, method="DOP853", rtol=1e-13, atol=1e-13, t_eval=t_eval)
    assert ours.y.shape == (4, t_eval.size)
    assert np.max(np.abs(ours.y[0] - ref.y[0])) < 1e-5


def test_dopri_rejects_steps_and_counts_evaluations():
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    sol = dopri45(model.rhs, (0, 30), model.resting_state(), rtol=1e-3, atol=1e-3)
    assert sol.n_rejected > 0  # the spike upstroke forces rejections at loose tolerance
    # 1 (f0) + 1 (initial-step probe) + 6 per attempted step (FSAL saves the 7th)
    assert sol.nfev == 2 + 6 * (sol.n_accepted + sol.n_rejected)


def test_dopri_batch_matches_single_trials_within_tolerance():
    amps = np.array([5.0, 10.0, 20.0])
    batch = simulate(HodgkinHuxley(stimulus=StepCurrent(amps, t_on=1.0)), (0, 40),
                     method="dopri45", rtol=1e-9, atol=1e-9, t_eval=np.linspace(0, 40, 401))
    for k, a in enumerate(amps):
        single = simulate(HodgkinHuxley(stimulus=StepCurrent(a, t_on=1.0)), (0, 40),
                          method="dopri45", rtol=1e-9, atol=1e-9, t_eval=np.linspace(0, 40, 401))
        np.testing.assert_allclose(batch.y[:, k], single.y, atol=1e-4)


# ---------------------------------------------------------------------------
# All methods through simulate()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_every_method_gives_the_same_spike_times(method):
    model = HodgkinHuxley(stimulus=StepCurrent(10.0, t_on=5.0, t_off=60.0))
    reference = simulate(model, (0, 70), dt=0.005)
    st_ref = spike_times(reference.t, reference.y[0])
    if method == "rk4":
        sol = simulate(model, (0, 70), method=method, dt=0.01)
    else:
        sol = simulate(model, (0, 70), method=method, rtol=1e-8, atol=1e-8,
                       t_eval=np.round(np.arange(0, 70.0001, 0.005), 10))
    st = spike_times(sol.t, sol.y[0])
    assert len(st) == len(st_ref) == 4
    np.testing.assert_allclose(st, st_ref, atol=1e-3)


def test_simulate_t_eval_across_breakpoints():
    model = HodgkinHuxley(stimulus=StepCurrent(10.0, t_on=2.5, t_off=7.5))
    t_eval = np.linspace(0, 10, 41)  # includes both breakpoints
    sol = simulate(model, (0, 10), method="dopri45", t_eval=t_eval)
    np.testing.assert_array_equal(sol.t, t_eval)
    assert sol.y.shape == (4, 41)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
def test_analytic_jacobian_matches_finite_differences():
    model = HodgkinHuxley(stimulus=ConstantCurrent(10.0))
    sol = simulate(model, (0, 20), dt=0.01, store_every=40)
    states = np.hstack([sol.y, [[-40.0, -55.0], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]])
    J = model.jacobian(0.0, states)
    for k in range(states.shape[1]):
        y = states[:, k]
        Jfd = np.empty((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = 1e-6 * max(1.0, abs(y[j]))
            Jfd[:, j] = (model.rhs(0, y + e) - model.rhs(0, y - e)) / (2 * e[j])
        np.testing.assert_allclose(J[k], Jfd, rtol=1e-6, atol=1e-6)


def test_rest_is_a_stable_focus():
    """Linearisation at rest: all eigenvalues in the left half-plane, one complex pair
    (damped subthreshold oscillations)."""
    model = HodgkinHuxley()
    lam = np.linalg.eigvals(model.jacobian(0, model.resting_state()))
    assert np.all(lam.real < 0)
    assert np.sum(np.abs(lam.imag) > 1e-9) == 2


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------
def test_bisect_threshold_on_known_step_function():
    thresholds = np.array([1.0, 2.5, 7.0])
    found = bisect_threshold(lambda a: a >= thresholds, 0.0, 10.0, rel_tol=1e-12)
    np.testing.assert_allclose(found, thresholds, rtol=1e-11)
    assert np.isinf(bisect_threshold(lambda a: a >= 20.0, 0.0, 10.0))


def test_fires_detects_crossings_per_trial():
    V = np.array([[-65, -10, 10, -60], [-65, -50, -40, -65.0]])
    np.testing.assert_array_equal(fires(V), [True, False])
