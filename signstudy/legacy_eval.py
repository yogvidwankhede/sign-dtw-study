"""FastDTW and legacy-pipeline evaluation (PREREG.md methods 2 and 3) at one setting: k=10, draw seed 0.

`legacy_fastdtw`: the original repository's logic on raw landmarks (normalised coordinates, per-hand detected frames,
hand sets must match, fastdtw default distance).  `fastdtw_r1`: FastDTW (radius 1, Euclidean) on the tuned F1 features,
for the accuracy/latency comparison with exact DTW.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import dataset as D
from . import experiment as X
from . import legacy as LG
from . import pipeline as P
from . import run as R


def _chunk(qs, refs, fn):
    return np.array([[fn(q, r) for r in refs] for q in qs], dtype=np.float32)


def _pairwise(queries, refs, fn, jobs=8):
    parts = np.array_split(np.arange(len(queries)), jobs * 4)
    res = Parallel(n_jobs=jobs)(delayed(_chunk)([queries[i] for i in p], refs, fn) for p in parts if len(p))
    return np.concatenate(res)


def run(split, n, k=10, seed=0):
    if split == "test":
        R.guard_test_split()
    t = P._tuned()["families"]["dtw"]
    signs = C.vocab(n)
    cond = X.build_condition(signs, "F1", t["missing"])
    sp = getattr(cond, split)
    order = X.orders_for(cond, seed)
    cols = np.array(sorted(i for s in signs for i in order[s][:k]))
    qmask = np.flatnonzero(sp.y < n)
    out = []

    # FastDTW on tuned F1 features (standardised if the DTW family chose that)
    from . import features as F
    sd = R._frame_std([cond.train.seqs[i] for i in cols]) if t["scale"] == "std" else np.float32(1.0)
    q = [F.dtw_sequence(sp.seqs[i]) / sd for i in qmask]
    r = [F.dtw_sequence(cond.train.seqs[i]) / sd for i in cols]
    t0 = time.time()
    d = _pairwise(q, r, LG.fastdtw_distance)
    secs = time.time() - t0
    sc = X.sign_min_scores(d, cond.train.y[cols], n)
    out.append(R._rows(sp, sc, sp.y[qmask], n, dict(method="fastdtw_r1", split=split, vocab=n, k=k, seed=seed,
                                                  fset="F1", missing=t["missing"], dominant=False, wall_s=round(secs, 1))))
    print("fastdtw_r1", round(secs), "s", flush=True)

    # legacy pipeline on raw landmarks
    vids_q = D.load_split(signs, split)
    keep = D.draw_queries(vids_q)
    vids_q = [vids_q[i] for i in keep]
    by_name_q = {v.name: v for v in vids_q}
    tr = {v.name: v for v in D.load_split(signs, "train")}
    leg_q = [LG.legacy_embedding(by_name_q[sp.videos[i].name]) for i in qmask]
    leg_r = [LG.legacy_embedding(tr[cond.train.videos[i].name]) for i in cols]
    t0 = time.time()
    d = _pairwise(leg_q, leg_r, LG.legacy_distance)
    secs = time.time() - t0
    sc = X.sign_min_scores(d, cond.train.y[cols], n)
    out.append(R._rows(sp, sc, sp.y[qmask], n, dict(method="legacy_fastdtw", split=split, vocab=n, k=k, seed=seed,
                                                  fset="legacy", missing="legacy", dominant=False, wall_s=round(secs, 1))))
    print("legacy", round(secs), "s", flush=True)
    df = pd.concat(out, ignore_index=True)
    R.save(df, "fastdtw_legacy", split)
    return df


if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]))
