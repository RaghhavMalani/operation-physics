"""EMA curriculum sampler: weights, gating, unlocking."""

import numpy as np
import pytest

from physcal.env.sampler import EMASampler, UniformSampler

CELLS = [(f, d) for f in ("dynamics", "energy", "kinematics") for d in (1, 2, 3, 4)]
CFG = {"alpha": 0.2, "weight_floor": 0.02, "prior_pass_rate": 0.5, "start_depths": [1, 2],
       "unlock_pass_rate": 0.5, "min_observations": 4}


def test_starts_at_depths_one_and_two():
    s = EMASampler(CELLS, CFG, seed=0)
    batch = s.next_batch(400)
    assert {d for _, d in batch} == {1, 2}
    assert len({f for f, _ in batch}) == 3


def test_weight_peaks_at_half_and_has_a_floor():
    s = EMASampler(CELLS, CFG, seed=0)
    s.p[("energy", 1)], s.p[("energy", 2)] = 0.5, 1.0
    w = s.weights()
    assert w[("energy", 1)] == pytest.approx(0.27) and w[("energy", 2)] == pytest.approx(0.02)


def test_ema_update_and_unlock():
    s = EMASampler(CELLS, CFG, seed=0)
    for _ in range(3):
        s.observe([(("energy", 2), [True] * 6 + [False] * 2)])
    assert ("energy", 3) not in s.unlocked  # too few observations
    s.observe([(("energy", 2), [True] * 8)])
    assert ("energy", 3) in s.unlocked and ("kinematics", 3) not in s.unlocked
    assert s.p[("energy", 2)] > 0.5
    # a family that keeps failing stays at depths 1-2
    for _ in range(20):
        s.observe([(("kinematics", 2), [False] * 8)])
    assert ("kinematics", 3) not in s.unlocked and s.p[("kinematics", 2)] < 0.05


def test_sampling_follows_weights():
    s = EMASampler(CELLS, CFG, seed=1)
    for c in list(s.unlocked):
        s.p[c] = 1.0  # mastered: floor weight only
    s.p[("dynamics", 1)] = 0.5
    counts = {c: 0 for c in s.unlocked}
    for c in s.next_batch(5000):
        counts[c] += 1
    assert counts[("dynamics", 1)] / 5000 == pytest.approx(0.27 / (0.27 + 5 * 0.02), abs=0.03)


def test_uniform_sampler():
    u = UniformSampler(CELLS, seed=0)
    assert len(set(u.next_batch(500))) == 12
