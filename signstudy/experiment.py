"""Shared experiment machinery: build a Condition (features, queries, reference orders) and score distance matrices."""
import hashlib
import pickle
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import dataset as D
from . import features as F


@dataclass
class Split:
    videos: list            # Video objects that produced a feature sequence
    seqs: list              # trimmed feature sequences (frames, dims)
    y: np.ndarray           # sign index in the vocabulary
    signer: np.ndarray
    failed: int = 0         # videos with no hand detected in any frame (counted as errors)
    failed_y: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    failed_signer: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


@dataclass
class Condition:
    signs: list
    fset: str
    missing: str
    dominant: bool
    train: Split
    val: Split
    test: Split
    orders: dict = field(default_factory=dict)   # seed -> {sign: [train idx in draw order]}


def _cache_path(signs, fset, missing, dominant):
    key = hashlib.sha1(("|".join(signs) + fset + missing + str(dominant)).encode()).hexdigest()[:12]
    return C.DATA / "features" / f"{fset}_{missing}_{int(dominant)}_{len(signs)}_{key}.pkl"


def _build_split(signs, split, fset, missing, dominant, queries_only):
    sign_idx = {s: i for i, s in enumerate(signs)}
    vids = D.load_split(signs, split, with_face=(fset == "F6"))
    if queries_only:
        keep = D.draw_queries(vids)
        vids = [vids[i] for i in keep]
    seqs, ok, bad = [], [], []
    for v in vids:
        x = F.video_features(v, fset, missing, dominant)
        (bad if x is None else ok).append(v)
        if x is not None:
            seqs.append(x)
    sp = Split(videos=ok, seqs=seqs, y=np.array([sign_idx[v.sign] for v in ok]),
               signer=np.array([v.signer for v in ok]), failed=len(bad),
               failed_y=np.array([sign_idx[v.sign] for v in bad], dtype=int),
               failed_signer=np.array([v.signer for v in bad]))
    for v in sp.videos:            # the raw landmarks are not needed again; free the memory
        v.left = v.right = v.pose = v.face = None
    return sp


def build_condition(signs, fset="F1", missing="zero", dominant=False, use_cache=True):
    path = _cache_path(signs, fset, missing, dominant)
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    D.check_signer_disjoint(signs)
    cond = Condition(signs, fset, missing, dominant,
                     train=_build_split(signs, "train", fset, missing, dominant, queries_only=False),
                     val=_build_split(signs, "val", fset, missing, dominant, queries_only=True),
                     test=_build_split(signs, "test", fset, missing, dominant, queries_only=True))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(cond, protocol=4))
    return cond


def orders_for(cond, seed):
    if seed not in cond.orders:
        cond.orders[seed] = D.reference_order(cond.train.videos, cond.signs, seed)
    return cond.orders[seed]


def sign_min_scores(dist, ref_sign, n_signs):
    """(Q, R) distances + reference sign ids -> (Q, n_signs) distance of the closest reference per sign (inf if none)."""
    out = np.full((dist.shape[0], n_signs), np.inf, dtype=np.float32)
    for s in range(n_signs):
        cols = np.flatnonzero(ref_sign == s)
        if len(cols):
            out[:, s] = dist[:, cols].min(axis=1)
    return out


def ranks_from_scores(scores, y):
    """Rank (1 = best) of the true sign; lower score is better. Ties are broken against the method (pessimistic)."""
    true = scores[np.arange(len(y)), y]
    return (scores < true[:, None]).sum(axis=1) + (scores == true[:, None]).sum(axis=1)


def per_query_table(split, ranks, extra):
    """Long-format rows, one per query, including failed videos (rank = inf -> counted wrong)."""
    import pandas as pd
    rows = pd.DataFrame({"signer": split.signer, "sign": split.y, "rank": ranks})
    if split.failed:
        rows = pd.concat([rows, pd.DataFrame({"signer": split.failed_signer, "sign": split.failed_y,
                                              "rank": np.iinfo(np.int32).max})], ignore_index=True)
    for k, v in extra.items():
        rows[k] = v
    rows["top1"] = rows["rank"] <= 1
    rows["top5"] = rows["rank"] <= 5
    return rows
