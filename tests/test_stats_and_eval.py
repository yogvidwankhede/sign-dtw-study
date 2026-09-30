import numpy as np
import pandas as pd

from signstudy import experiment as X
from signstudy import run as R
from signstudy import stats as S
from signstudy.experiment import Condition, Split


def _df(method, accs, seed=0, vocab=10, k=5, n=20):
    rows = []
    rng = np.random.default_rng(seed)
    for i, a in enumerate(accs):
        ok = rng.random(n) < a
        for j in range(n):
            rows.append(dict(method=method, vocab=vocab, k=k, fset="F1", missing="zero", dominant=False,
                             seed=seed, signer=f"s{i}", rank=1 if ok[j] else 2))
    return pd.DataFrame(rows)


def test_bootstrap_contains_point_and_orders():
    df = pd.concat([_df("a", [0.9] * 8), _df("b", [0.5] * 8)])
    ci = S.accuracy_ci(S.signer_table(df), B=2000)
    a = ci[ci.method == "a"].iloc[0]
    assert a.lo <= a.acc <= a.hi and a.acc > ci[ci.method == "b"].iloc[0].acc


def test_paired_difference_excludes_zero_for_large_gap():
    df = pd.concat([_df("a", [0.95] * 8), _df("b", [0.4] * 8)])
    d = S.paired_difference(S.signer_table(df), "a", "b", dict(vocab=10, k=5, fset="F1"), B=2000)
    assert d["lo"] > 0


def test_ranks_are_pessimistic_on_ties():
    scores = np.array([[1.0, 1.0, 2.0]])
    assert X.ranks_from_scores(scores, np.array([0]))[0] == 2
    assert X.ranks_from_scores(scores, np.array([2]))[0] == 3


def test_failed_videos_count_as_errors():
    split = Split(videos=[], seqs=[], y=np.array([0, 1]), signer=np.array(["a", "a"]), failed=1,
                  failed_y=np.array([0]), failed_signer=np.array(["a"]))
    scores = np.array([[0.0, 1.0], [1.0, 0.0]])
    df = R._rows(split, scores, split.y, 2, {"m": 1})
    assert len(df) == 3 and (df["rank"] <= 1).sum() == 2


def test_eval_distance_nearest_reference_and_nested_k():
    # 3 signs, 4 train refs each; query = exact copy of a ref, so 1-NN must be perfect for any k that includes it
    signs = ["a", "b", "c"]
    seqs = [np.full((4, 2), float(s * 10 + r), dtype=np.float32) for s in range(3) for r in range(4)]
    y = np.array([s for s in range(3) for _ in range(4)])
    from signstudy.dataset import Video
    vids = [Video(sign=signs[s], name=f"{s}-{r}") for s in range(3) for r in range(4)]
    train = Split(vids, seqs, y, np.array(["t"] * 12))
    q = Split([], [np.full((4, 2), float(s * 10), dtype=np.float32) for s in range(3)], np.arange(3), np.array(["q"] * 3))
    cond = Condition(signs, "F1", "zero", False, train, q, q)
    from signstudy.dtwlib import dtw_matrix
    dist = dtw_matrix(q.seqs, seqs)
    df = R.eval_distance(cond, "val", dist, "dtw", [3], ["full", 1], [0, 1])
    full = df[df.k == "full"]
    assert (full["rank"] == 1).all()
    assert set(df[df.k == 1].seed) == {0, 1}
