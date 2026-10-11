"""Tests of the experiment library: equilibria, bifurcation points, thresholds, statistics,
refractoriness and the planar reduction. Reference values are the ones the scripts report;
they agree with the published HH literature (Hopf at ~9.78 and ~154.5 uA/cm^2, rheobase ~2.24)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hh.config import DEFAULT_PARAMS
from hh.experiments import (
    ap_metrics,
    equilibrium,
    fi_curve,
    firing_rate,
    fit_h_n_relation,
    hopf_interval,
    oscillation_frequency,
    passive_ceiling,
    pulse_threshold,
    refractory_curve,
    simulate,
    step_threshold,
)
from hh.model import HodgkinHuxley, RinzelReduced, rates, steady_state_current
from hh.stimuli import StepCurrent


def test_steady_state_current_is_monotone_so_equilibrium_is_unique():
    V = np.linspace(-120, 60, 20001)
    assert np.all(np.diff(steady_state_current(V)) > 0)


def test_equilibrium_at_zero_current_is_rest():
    np.testing.assert_allclose(equilibrium(DEFAULT_PARAMS, 0.0), HodgkinHuxley().resting_state(), atol=1e-10)


def test_hopf_points_match_literature():
    lo, hi = hopf_interval(DEFAULT_PARAMS, np.arange(0.0, 200.0, 0.5))
    assert lo == pytest.approx(9.7754, abs=1e-3)
    assert hi == pytest.approx(154.5224, abs=1e-3)


def test_firing_rate_and_oscillation_frequency_on_synthetic_signals():
    t = np.arange(0.0, 1000.0, 0.05)
    spikes = 60 * np.sin(2 * np.pi * t / 10.0) - 20  # crosses 0 mV, 100 Hz
    ripple = 5 * np.sin(2 * np.pi * t / 4.0) - 40  # never reaches 0 mV, 250 Hz
    assert firing_rate(t, spikes, 100.0) == pytest.approx(100.0, rel=1e-4)
    assert firing_rate(t, ripple, 100.0) == 0.0
    assert oscillation_frequency(t, ripple, 100.0) == pytest.approx(250.0, rel=1e-4)
    assert oscillation_frequency(t, np.full_like(t, -65.0), 100.0) == 0.0


def test_fi_curve_known_points():
    res = fi_curve(np.array([5.0, 10.0, 20.0]), t_end=400.0, t_transient=150.0)
    assert res.spike_rate[0] == 0.0
    assert res.spike_rate[1] == pytest.approx(68.3, abs=0.2)  # ISI 14.64 ms
    assert res.spike_rate[2] == pytest.approx(86.5, abs=0.3)


def test_pulse_threshold_and_passive_ceiling():
    th = pulse_threshold(width=0.5, rel_tol=1e-8)
    assert th == pytest.approx(13.27512356, rel=1e-6)
    ceiling = passive_ceiling(DEFAULT_PARAMS, 0.5, hi=1000.0, rel_tol=1e-4)
    assert ceiling > 10 * th  # thresholds near rest are far from the passive regime


def test_rheobase():
    assert step_threshold(t_end=60.0, hi=10.0, rel_tol=1e-4) == pytest.approx(2.2403, abs=2e-3)


def test_ap_metrics_of_standard_spike():
    model = HodgkinHuxley(stimulus=StepCurrent(1.5 * 13.2751, 1.0, 1.5))
    sol = simulate(model, (0.0, 40.0), dt=0.005)
    mt = ap_metrics(sol.t, sol.y[0], model.resting_potential())
    assert mt["peak"] == pytest.approx(39.3, abs=0.1)
    assert mt["half_width"] == pytest.approx(1.458, abs=0.01)
    assert mt["ahp_min"] == pytest.approx(-76.17, abs=0.05)
    assert np.isnan(ap_metrics(sol.t, np.full_like(sol.t, -65.0), -65.0)["peak"])


def test_refractory_curve_shape():
    res = refractory_curve(intervals=np.array([1.0, 3.0, 9.0, 12.0, 40.0]), rel_tol=1e-3)
    # 1.0 ms is dropped: the conditioning spike has not yet crossed 0 mV
    np.testing.assert_allclose(res.intervals, [3.0, 9.0, 12.0, 40.0])
    ratio = res.thresholds / res.resting_threshold
    assert np.isinf(ratio[0])  # absolute refractory
    assert ratio[1] > ratio[2] > 1.5  # relative, recovering
    assert ratio[3] == pytest.approx(1.0, abs=0.01)  # recovered


def test_h_n_relation_matches_rinzel():
    h0, k, rms = fit_h_n_relation()
    assert h0 == pytest.approx(0.89, abs=0.03)
    assert k == pytest.approx(1.1, abs=0.05)
    assert rms < 0.02


def test_reduced_model_n_equation_is_the_full_one():
    red = RinzelReduced()
    V, n = np.linspace(-80, 40, 50), np.linspace(0.2, 0.8, 50)
    a, b = rates("n", V)
    np.testing.assert_allclose(red.dn(V, n), a * (1 - n) - b * n)
    assert red.rhs(0.0, np.vstack([V, n])).shape == (2, 50)


def test_temperature_speeds_up_the_spike():
    warm = replace(DEFAULT_PARAMS, temperature=16.3)
    t_peak = []
    for p in (DEFAULT_PARAMS, warm):
        sol = simulate(HodgkinHuxley(params=p, stimulus=StepCurrent(30.0, 1.0, 1.5)), (0, 20), dt=0.002)
        t_peak.append(sol.t[np.argmax(sol.y[0])])
    assert t_peak[1] < t_peak[0]
