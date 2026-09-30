import numpy as np
from tslearn.metrics import dtw as ts_dtw

from signstudy.dtwlib import dtw_matrix, euclid_matrix


def _seqs(rng, n, d=6):
    return [rng.normal(size=(rng.integers(5, 15), d)).astype(np.float32) for _ in range(n)]


def test_matches_tslearn_unconstrained():
    rng = np.random.default_rng(0)
    q, r = _seqs(rng, 4), _seqs(rng, 5)
    mine = dtw_matrix(q, r)
    ref = np.array([[ts_dtw(a, b) for b in r] for a in q])
    assert np.allclose(mine, ref, rtol=1e-4, atol=1e-4)


def test_matches_tslearn_band_for_equal_length():
    rng = np.random.default_rng(1)
    q = [rng.normal(size=(12, 4)).astype(np.float32) for _ in range(3)]
    r = [rng.normal(size=(12, 4)).astype(np.float32) for _ in range(3)]
    mine = dtw_matrix(q, r, radius=3)
    ref = np.array([[ts_dtw(a, b, global_constraint="sakoe_chiba", sakoe_chiba_radius=3) for b in r] for a in q])
    assert np.allclose(mine, ref, rtol=1e-4, atol=1e-4)


def test_self_distance_zero_and_symmetry():
    rng = np.random.default_rng(2)
    s = _seqs(rng, 4)
    d = dtw_matrix(s, s)
    assert np.allclose(np.diag(d), 0, atol=1e-5) and np.allclose(d, d.T, atol=1e-4)


def test_band_never_below_unconstrained():
    rng = np.random.default_rng(3)
    s = _seqs(rng, 4)
    assert (dtw_matrix(s, s, radius=2) >= dtw_matrix(s, s) - 1e-4).all()


def test_euclid_matrix():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(3, 7, 2)), rng.normal(size=(4, 7, 2))
    ref = np.linalg.norm(a.reshape(3, 1, -1) - b.reshape(1, 4, -1), axis=-1)
    assert np.allclose(euclid_matrix(a, b), ref, atol=1e-4)
