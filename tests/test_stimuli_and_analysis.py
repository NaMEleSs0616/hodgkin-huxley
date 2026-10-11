"""Tests of the stimulus protocols and spike detection."""

from __future__ import annotations

import numpy as np
import pytest

from hh.experiments import spike_times
from hh.stimuli import ConstantCurrent, PulsePair, StepCurrent


def test_step_current_is_on_for_half_open_interval():
    s = StepCurrent(5.0, t_on=10.0, t_off=20.0)
    assert [float(s(t)) for t in (9.999, 10.0, 19.999, 20.0)] == [0.0, 5.0, 5.0, 0.0]
    assert s.breakpoints == (10.0, 20.0)
    assert StepCurrent(1.0, t_on=3.0).breakpoints == (3.0,)


def test_step_current_batches_over_amplitudes():
    s = StepCurrent(np.array([1.0, 2.0, 3.0]), t_on=1.0)
    np.testing.assert_array_equal(s(0.0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(s(2.0), [1.0, 2.0, 3.0])


def test_constant_current():
    assert float(ConstantCurrent(4.0)(123.0)) == 4.0
    assert ConstantCurrent(4.0).breakpoints == ()


def test_pulse_pair_batches_over_intervals():
    p = PulsePair(amplitude=10.0, width=1.0, t_first=5.0, interval=np.array([3.0, 8.0]))
    np.testing.assert_array_equal(p(5.5), [10.0, 10.0])  # first pulse, both trials
    np.testing.assert_array_equal(p(8.5), [10.0, 0.0])  # second pulse, trial 0 only
    np.testing.assert_array_equal(p(13.5), [0.0, 10.0])  # second pulse, trial 1 only
    assert p.breakpoints == (5.0, 6.0, 8.0, 9.0, 13.0, 14.0)


def test_spike_times_exact_for_piecewise_linear_signal():
    t = np.arange(0.0, 10.0, 1.0)
    V = np.array([-60, -20, 20, 0, -60, -60, -10, 30, -50, -60], dtype=float)
    # Upward crossings of 0: between t=1 and 2 (at 1.5) and between t=6 and 7 (at 6.25).
    np.testing.assert_allclose(spike_times(t, V), [1.5, 6.25])


def test_spike_times_on_sine():
    t = np.linspace(0, 10 * np.pi, 20001)
    st = spike_times(t, np.sin(t - 0.3))
    np.testing.assert_allclose(st, 0.3 + 2 * np.pi * np.arange(0, 5), atol=1e-6)


def test_no_spikes_when_subthreshold():
    assert spike_times(np.arange(5.0), np.full(5, -65.0)).size == 0
