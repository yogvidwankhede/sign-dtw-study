"""Evaluate methods. `--split val` is for development and tuning; `--split test` is refused unless the
pre-registration tag exists, is an ancestor of HEAD, and the working tree is clean."""
import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import dtwlib as L
from . import experiment as X
from . import features as F
from . import train as T


def git(*args):
    return subprocess.run(["git", "-C", str(C.ROOT), *args], capture_output=True, text=True).stdout.strip()


def guard_test_split():
    if not git("tag", "--list", "prereg-v1"):
        sys.exit("refusing to write test results: tag prereg-v1 does not exist")
    if subprocess.run(["git", "-C", str(C.ROOT), "merge-base", "--is-ancestor", "prereg-v1", "HEAD"]).returncode:
        sys.exit("prereg-v1 is not an ancestor of HEAD")
    if git("status", "--porcelain", "--", "signstudy", "PREREG.md", "tests"):
        sys.exit("uncommitted changes in signstudy/, tests/ or PREREG.md: commit them first")


def provenance():
    import cv2
    import mediapipe
    import torch
    import tslearn
    return dict(commit=git("rev-parse", "HEAD"), prereg_diff=git("diff", "--stat", "prereg-v1", "HEAD", "--", "signstudy"),
                python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, tslearn=tslearn.__version__,
                mediapipe=mediapipe.__version__, opencv=cv2.__version__, machine=platform.platform(),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------- scoring helpers -------------------------------------------------------------------------------------

def _rows(split, scores, y, n_vocab, meta):
    """Query rows for vocabulary n_vocab from a (Q, n_vocab) distance-like matrix (lower = better)."""
    ranks = X.ranks_from_scores(scores, y)
    preds = scores.argmin(1)
    df = pd.DataFrame({"signer": split.signer[split.y < n_vocab], "sign": y, "rank": ranks, "pred": preds})
    bad = split.failed_y < n_vocab
    if bad.any():
        df = pd.concat([df, pd.DataFrame({"signer": split.failed_signer[bad], "sign": split.failed_y[bad],
                                          "rank": np.iinfo(np.int32).max, "pred": -1})], ignore_index=True)
    for k, v in meta.items():
        df[k] = v
    return df


def _ref_cols(cond, n_vocab, k, seed):
    order = X.orders_for(cond, seed)
    cols = []
    for s in cond.signs[:n_vocab]:
        idx = order[s]
        cols.extend(idx if k == "full" else idx[:k])
    # `order` holds indices into cond.train.videos == positions in cond.train.seqs
    return np.array(sorted(cols), dtype=int)


def eval_distance(cond, split_name, dist, label, vocabs, ks, seeds, extra=None):
    """1-NN style evaluation from a (queries, all-train-refs) distance matrix."""
    split = getattr(cond, split_name)
    out = []
    for n in vocabs:
        qmask = split.y < n
        for k in ks:
            for seed in (seeds if k != "full" else seeds[:1]):
                cols = _ref_cols(cond, n, k, seed)
                sc = X.sign_min_scores(dist[qmask][:, cols], cond.train.y[cols], n)
                meta = dict(method=label, split=split_name, vocab=n, k=k, seed=seed if k != "full" else -1,
                            fset=cond.fset, missing=cond.missing, dominant=cond.dominant, **(extra or {}))
                out.append(_rows(split, sc, split.y[qmask], n, meta))
    return pd.concat(out, ignore_index=True)


def dtw_dist(cond, split_name, radius):
    split = getattr(cond, split_name)
    q = [F.dtw_sequence(x) for x in split.seqs]
    r = [F.dtw_sequence(x) for x in cond.train.seqs]
    return L.dtw_matrix(q, r, radius)


def euclid_dist(cond, split_name):
    split = getattr(cond, split_name)
    q = np.stack([F.resample(x) for x in split.seqs])
    r = np.stack([F.resample(x) for x in cond.train.seqs])
    return L.euclid_matrix(q, r)


def _frame_std(seqs):
    """Per-dimension standard deviation over all frames of `seqs` (streamed; floor 1e-3 -> 1)."""
    n, s1, s2 = 0, 0.0, 0.0
    for x in seqs:
        x = x.astype(np.float64)
        n += len(x)
        s1 = s1 + x.sum(0)
        s2 = s2 + (x * x).sum(0)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0))
    return np.where(sd < 1e-3, 1.0, sd).astype(np.float32)


def eval_scaled(cond, split_name, label, vocabs, ks, seeds, kind="dtw", radius=-1):
    """DTW or Euclidean 1-NN after dividing every feature dimension by its standard deviation over the selected
    references (the same standardisation the learned models get; centring cancels in differences)."""
    split = getattr(cond, split_name)
    out = []
    for n in vocabs:
        qmask = np.flatnonzero(split.y < n)
        for k in ks:
            for seed in (seeds if k != "full" else seeds[:1]):
                cols = _ref_cols(cond, n, k, seed)
                sd = _frame_std([cond.train.seqs[i] for i in cols])
                if kind == "dtw":
                    q = [F.dtw_sequence(split.seqs[i]) / sd for i in qmask]
                    r = [F.dtw_sequence(cond.train.seqs[i]) / sd for i in cols]
                    d = L.dtw_matrix(q, r, radius)
                else:
                    q = np.stack([F.resample(split.seqs[i]) / sd for i in qmask])
                    r = np.stack([F.resample(cond.train.seqs[i]) / sd for i in cols])
                    d = L.euclid_matrix(q, r)
                sc = X.sign_min_scores(d, cond.train.y[cols], n)
                meta = dict(method=label, split=split_name, vocab=n, k=k, seed=seed if k != "full" else -1,
                            fset=cond.fset, missing=cond.missing, dominant=cond.dominant,
                            **({"radius": radius} if kind == "dtw" else {}))
                out.append(_rows(split, sc, split.y[qmask], n, meta))
    return pd.concat(out, ignore_index=True)


def eval_prototypes(cond, split_name, vocabs, ks, seeds, radius=-1, max_iter=30, standardise=False, label="dba_proto"):
    """DBA prototype per sign, nearest prototype by DTW."""
    from tslearn.barycenters import dtw_barycenter_averaging
    split = getattr(cond, split_name)
    q = [F.dtw_sequence(x) for x in split.seqs]
    n_all = max(vocabs)
    out = []
    for k in ks:
        for seed in (seeds if k != "full" else seeds[:1]):
            order = X.orders_for(cond, seed)
            protos = []
            sd = (_frame_std([cond.train.seqs[i] for i in _ref_cols(cond, n_all, k, seed)])
                  if standardise else np.float32(1.0))
            qs = [x / sd for x in q]
            for s in cond.signs[:n_all]:
                idx = order[s] if k == "full" else order[s][:k]
                seqs = [F.dtw_sequence(cond.train.seqs[i]) / sd for i in idx]
                protos.append(seqs[0] if len(seqs) == 1 else
                              dtw_barycenter_averaging(seqs, max_iter=max_iter).astype(np.float32))
            dist = L.dtw_matrix(qs, protos, radius)
            for n in vocabs:
                qmask = split.y < n
                meta = dict(method=label, split=split_name, vocab=n, k=k, seed=seed if k != "full" else -1,
                            fset=cond.fset, missing=cond.missing, dominant=cond.dominant)
                out.append(_rows(split, dist[qmask][:, :n], split.y[qmask], n, meta))
    return pd.concat(out, ignore_index=True)


def eval_learned(cond, split_name, model, vocabs, ks, seeds, lr=1e-3, aug=False, epochs=20, tag=None):
    split = getattr(cond, split_name)
    qx = np.stack([F.resample(x) for x in split.seqs])
    trx = np.stack([F.resample(x) for x in cond.train.seqs])
    out, timings = [], []
    for n in vocabs:
        qmask = split.y < n
        for k in ks:
            for seed in (seeds if k != "full" else seeds[:1]):
                cols = _ref_cols(cond, n, k, seed)
                st = T.steps_for(len(cols), epochs=epochs)
                (logits,), secs, _ = T.fit_predict(model, trx[cols], cond.train.y[cols], [qx[qmask]], n, seed, st,
                                                   lr=lr, aug=aug)
                meta = dict(method=(tag or model), split=split_name, vocab=n, k=k, seed=seed if k != "full" else -1,
                            fset=cond.fset, missing=cond.missing, dominant=cond.dominant)
                out.append(_rows(split, -logits, split.y[qmask], n, meta))
                timings.append(dict(method=(tag or model), vocab=n, k=k, seed=seed, steps=st, train_s=round(secs, 1)))
                print(model, n, k, seed, "steps", st, round(secs, 1), "s", flush=True)
    return pd.concat(out, ignore_index=True), pd.DataFrame(timings)


def save(df, name, split):
    path = C.RESULTS / "raw" / split / f"{name}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    (path.parent / f"{name}.provenance.json").write_text(json.dumps(provenance(), indent=1))
    return path


def accuracy(df):
    """Quick summary used during development: top-1/top-5 per method/vocab/k averaged over seeds."""
    g = df.groupby(["method", "vocab", "k", "seed"]).agg(top1=("rank", lambda r: (r <= 1).mean()),
                                                         top5=("rank", lambda r: (r <= 5).mean())).reset_index()
    return g.groupby(["method", "vocab", "k"]).agg(top1=("top1", "mean"), top5=("top5", "mean"),
                                                   seeds=("seed", "nunique")).reset_index()
