"""End-to-end check on real extracted landmarks: our DTW 1-NN predictions equal an independent tslearn brute force.
Skipped when the landmark cache is absent. Uses the validation split only."""
import numpy as np
import pytest

from signstudy import config as C
from signstudy import experiment as X
from signstudy import features as F
from signstudy import run as R

pytestmark = pytest.mark.skipif(not (C.LANDMARKS / C.vocab(10)[-1] / "val.npz").exists(), reason="landmarks not extracted")


def test_dtw_predictions_match_tslearn_on_real_data():
    from tslearn.metrics import dtw as ts_dtw
    cond = X.build_condition(C.vocab(10), "F1", "drop")
    sp = cond.val
    cols = R._ref_cols(cond, 10, 5, 0)
    q = [F.dtw_sequence(x) for x in sp.seqs[:40]]
    r = [F.dtw_sequence(cond.train.seqs[i]) for i in cols]
    mine = R.L.dtw_matrix(q, r, -1)
    ref = np.array([[ts_dtw(a, b) for b in r] for a in q])
    assert np.abs(mine - ref).max() < 1e-3
    y = cond.train.y[cols]
    assert (y[mine.argmin(1)] == y[ref.argmin(1)]).all()
