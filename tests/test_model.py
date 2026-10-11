"""Tests of the model equations: rate functions, steady states, rest, and invariants."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hh.config import DEFAULT_PARAMS
from hh.model import (
    GATES,
    HodgkinHuxley,
    alpha_m,
    alpha_n,
    ionic_currents,
    steady_state,
    time_constant,
)
from hh.stimuli import ConstantCurrent

V_GRID = np.linspace(-120.0, 60.0, 3601)


# ---------------------------------------------------------------------------
# Removable singularities
# ---------------------------------------------------------------------------
def naive_alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))


def naive_alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))


@pytest.mark.parametrize(
    "safe, naive, V0, limit",
    [(alpha_m, naive_alpha_m, -40.0, 1.0), (alpha_n, naive_alpha_n, -55.0, 0.1)],
)
class TestRemovableSingularity:
    def test_value_at_singular_point_is_the_limit(self, safe, naive, V0, limit):
        # l'Hopital: a(V-V0)/(1-exp(-(V-V0)/k)) -> a*k as V -> V0
        assert safe(V0) == pytest.approx(limit, rel=1e-15)

    def test_matches_textbook_formula_away_from_singularity(self, safe, naive, V0, limit):
        V = V_GRID[np.abs(V_GRID - V0) > 0.05]
        np.testing.assert_allclose(safe(V), naive(V), rtol=1e-11)

    def test_smooth_through_singularity(self, safe, naive, V0, limit):
        # Near u = 0, a*k*u/(e^u - 1) = a*k*(1 - u/2 + ...) with u = -(V-V0)/k, so d(alpha)/dV = a/2.
        # A centred difference across V0 must reproduce that slope (no kink, no blow-up).
        a_coeff = limit / 10.0
        d = 1e-4
        slope = (safe(V0 + d) - safe(V0 - d)) / (2 * d)
        assert slope == pytest.approx(a_coeff / 2.0, rel=1e-6)

    def test_safe_formula_is_accurate_where_naive_cancels(self, safe, naive, V0, limit):
        # 1e-13 mV from V0 the naive formula is visibly wrong; the safe one equals the limit.
        assert safe(V0 + 1e-13) == pytest.approx(limit, rel=1e-12)
        assert abs(naive(V0 + 1e-13) - limit) / limit > 1e-4


# ---------------------------------------------------------------------------
# Steady states and time constants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gate", GATES)
def test_steady_states_are_probabilities(gate):
    x = steady_state(gate, V_GRID)
    assert np.all((x > 0.0) & (x < 1.0))


@pytest.mark.parametrize("gate", GATES)
def test_time_constants_positive_and_finite(gate):
    tau = time_constant(gate, V_GRID)
    assert np.all(np.isfinite(tau) & (tau > 0.0))


def test_activation_increases_and_inactivation_decreases_with_voltage():
    assert np.all(np.diff(steady_state("m", V_GRID)) > 0)
    assert np.all(np.diff(steady_state("n", V_GRID)) > 0)
    assert np.all(np.diff(steady_state("h", V_GRID)) < 0)


def test_temperature_scales_rates_but_not_steady_states():
    warm = replace(DEFAULT_PARAMS, temperature=16.3)  # 10 degC above reference -> phi = Q10 = 3
    assert DEFAULT_PARAMS.phi == 1.0
    assert warm.phi == pytest.approx(3.0)
    for gate in GATES:
        np.testing.assert_allclose(time_constant(gate, V_GRID, warm.phi),
                                   time_constant(gate, V_GRID) / 3.0, rtol=1e-14)


# ---------------------------------------------------------------------------
# Resting state
# ---------------------------------------------------------------------------
def test_resting_state_matches_standard_values():
    V, m, n, h = HodgkinHuxley().resting_state()
    assert V == pytest.approx(-65.0, abs=0.01)
    # Standard textbook initial conditions for the -65 mV convention.
    assert m == pytest.approx(0.0529, abs=2e-4)
    assert n == pytest.approx(0.3177, abs=2e-4)
    assert h == pytest.approx(0.5961, abs=2e-4)


@pytest.mark.parametrize("I_hold", [0.0, 2.0, -5.0])
def test_resting_state_is_a_fixed_point(I_hold):
    model = HodgkinHuxley(stimulus=ConstantCurrent(I_hold))
    y_rest = model.resting_state(I_hold)
    assert np.max(np.abs(model.rhs(0.0, y_rest))) < 1e-12


def test_currents_balance_at_rest():
    """At rest the Na influx, K efflux and leak sum to zero, though none is zero alone."""
    y_rest = HodgkinHuxley().resting_state()
    I_Na, I_K, I_L = ionic_currents(y_rest)
    assert I_Na < 0 < I_K  # Na flows in (inward = negative), K flows out
    assert I_Na + I_K + I_L == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Vector field structure
# ---------------------------------------------------------------------------
def test_unit_cube_is_forward_invariant_for_gates():
    """On x = 0, dx/dt = alpha > 0; on x = 1, dx/dt = -beta < 0, for every voltage.

    So the flow points into [0, 1] on both faces and gating variables can never leave
    it: the exact solution respects the probabilistic meaning of x.
    """
    model = HodgkinHuxley()
    N = V_GRID.size
    at_zero = model.rhs(0.0, np.vstack([V_GRID, np.zeros((3, N))]))
    at_one = model.rhs(0.0, np.vstack([V_GRID, np.ones((3, N))]))
    assert np.all(at_zero[1:] > 0.0)
    assert np.all(at_one[1:] < 0.0)


def test_rhs_batch_equals_columnwise():
    rng = np.random.default_rng(0)
    Y = np.vstack([rng.uniform(-90, 40, 25), rng.uniform(0, 1, (3, 25))])
    I = rng.uniform(-5, 20, 25)
    batch = HodgkinHuxley(stimulus=ConstantCurrent(I)).rhs(0.0, Y)
    cols = np.column_stack(
        [HodgkinHuxley(stimulus=ConstantCurrent(I[k])).rhs(0.0, Y[:, k]) for k in range(25)]
    )
    np.testing.assert_allclose(batch, cols, rtol=1e-15)


def test_each_current_pulls_V_toward_its_reversal_potential():
    """With only one conductance active, dV/dt has the sign of (E_ion - V)."""
    p = DEFAULT_PARAMS
    for g_field, E in [("g_Na", p.E_Na), ("g_K", p.E_K), ("g_L", p.E_L)]:
        only = replace(p, **{"g_Na": 0.0, "g_K": 0.0, "g_L": 0.0, g_field: 1.0})
        model = HodgkinHuxley(params=only)
        y = np.vstack([V_GRID, np.full((3, V_GRID.size), 0.5)])
        dV = model.rhs(0.0, y)[0]
        away = np.abs(V_GRID - E) > 1e-9
        assert np.all(np.sign(dV[away]) == np.sign(E - V_GRID[away]))


def test_rhs_rejects_mismatched_batch_shapes():
    model = HodgkinHuxley(stimulus=ConstantCurrent(np.array([1.0, 2.0, 3.0])))
    with pytest.raises(ValueError, match="one column per trial"):
        model.rhs(0.0, model.resting_state())
