import os
import subprocess
import sys

import numpy as np
import pytest

from signstudy import config as C
from signstudy import experiment as X
from signstudy import extra as E
from signstudy.dataset import Video
from signstudy.experiment import Condition, Split


def test_knn_scores_kappa_one_equals_minimum_and_mean_of_smallest():
    dist = np.array([[1.0, 5.0, 2.0, 9.0]], dtype=np.float32)
    sign = np.array([0, 0, 1, 1])
    assert np.allclose(X.sign_knn_scores(dist, sign, 2, 1), [[1.0, 2.0]])
    assert np.allclose(X.sign_knn_scores(dist, sign, 2, 3), [[3.0, 5.5]])      # kappa capped at the 2 references each sign has


def _cond(means, labels):
    seqs = [np.tile(np.asarray(m, dtype=np.float32), (5, 1)) + 0.01 * np.arange(5)[:, None] for m in means]
    vids = [Video(sign=str(l), name=f"{i}") for i, l in enumerate(labels)]
    sp = Split(vids, seqs, np.array(labels), np.array(["t"] * len(labels)))
    return Condition(["0", "1"], "F1", "zero", False, sp, sp, sp)


def test_fisher_weights_prefer_discriminative_dimensions_and_are_ones_for_single_references():
    rng = np.random.default_rng(0)
    means = [[c * 5 + rng.normal(0, 0.1), rng.normal(0, 1)] for c in (0, 1) for _ in range(6)]
    cond = _cond(means, [0] * 6 + [1] * 6)
    cols = np.arange(12)
    w = E.fisher_weights(cond, cols, np.ones(2, dtype=np.float32))
    assert w[0] > w[1] and np.isclose(w.mean(), 1.0, atol=0.5)
    single = E.fisher_weights(cond, np.array([0, 6]), np.ones(2, dtype=np.float32))
    assert np.allclose(single, 1.0)


def test_dataset_switch_changes_paths_and_grids():
    code = ("from signstudy import config as C;"
            "print(C.RESULTS.name, C.PREREG_TAG, C.MACRO_PREFIX, C.VOCAB_SIZES[-1], 20 in C.REFS_PER_SIGN)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=C.ROOT,
                         env={**os.environ, "SIGNSTUDY_DATASET": "aslc"}).stdout.split()
    assert out == ["results_aslc", "prereg-v2", "cz", "300", "False"]


def test_minirocket_runs_on_tiny_data():
    pytest.importorskip("aeon")
    rng = np.random.default_rng(0)
    cond = _cond([[c * 3 + rng.normal(0, 0.1), rng.normal(0, 0.1)] for c in (0, 1) for _ in range(4)], [0] * 4 + [1] * 4)
    # Condition's reference order needs real Video names; reuse them as the test split as well
    df = E.eval_minirocket(cond, "val", [2], ["full"], [0], n_kernels=84)
    assert len(df) == 8 and set(df.columns) >= {"rank", "pred", "signer"}
