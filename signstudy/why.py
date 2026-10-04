"""Descriptive 'why' analyses (PREREG_ADDENDUM.md Part B5, exploratory). PopSign only. Writes results/why.json and tables."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import dataset as D
from . import experiment as X
from . import features as F
from . import pipeline as P
from . import run as R
from . import train as T

BLOCK = 210   # F1 per-hand block


def slot_usage(signs, split="test"):
    """Per signer: share of hand-present frames with only the left slot, only the right slot, or both."""
    rows = {}
    for v in D.load_split(signs, split):
        l, r = ~np.isnan(v.left[:, 0, 0]), ~np.isnan(v.right[:, 0, 0])
        c = rows.setdefault(v.signer, np.zeros(3))
        c += [int((l & ~r).sum()), int((r & ~l).sum()), int((l & r).sum())]
    out = pd.DataFrame({s: c / max(c.sum(), 1) for s, c in rows.items()}, index=["left_only", "right_only", "both"]).T
    out.index.name = "signer"
    return out.reset_index()


def per_signer_acc(method, vocab, k):
    d = pd.read_csv(C.RESULTS / "raw" / "test" / f"{method}.csv.gz")
    d["k"] = d["k"].astype(str)
    d = d[(d.vocab == vocab) & (d.k == str(k)) & (d.fset == "F1")]
    return d.groupby("signer").apply(lambda g: (g["rank"] <= 1).mean(), include_groups=False)


def swap_slots(x):
    return np.concatenate([x[:, BLOCK:2 * BLOCK], x[:, :BLOCK]], axis=1)


def slot_swap_test(n, ks=(10, "full")):
    """Accuracy of DTW 1-NN and a CNN when the left/right hand slots of the test queries are swapped."""
    t = P._tuned()["families"]
    res = []
    cond = X.build_condition(C.vocab(n), "F1", t["dtw"]["missing"])
    sw = X.Condition(**{**cond.__dict__})
    sw.test = X.Split(**{**cond.test.__dict__, "seqs": [swap_slots(x) for x in cond.test.seqs]})
    for k in ks:
        P_ks = [k]
        for name, c in (("normal", cond), ("swapped", sw)):
            df = P._dtw_family(c, "test", t["dtw"], [n], ks=P_ks)
            res.append(dict(method="dtw1nn", queries=name, k=str(k), top1=float((df["rank"] <= 1).mean())))
    hp = t["cnn1d"]
    cond = X.build_condition(C.vocab(n), "F1", hp["missing"])
    trx = np.stack([F.resample(x) for x in cond.train.seqs])
    q = np.stack([F.resample(x) for x in cond.test.seqs])
    qs = np.stack([F.resample(swap_slots(x)) for x in cond.test.seqs])
    for k in ks:
        accs = {"normal": [], "swapped": []}
        for seed in (list(C.SEEDS) if k != "full" else [0]):
            cols = R._ref_cols(cond, n, k, seed)
            (lg, lgs), _, _ = T.fit_predict("cnn1d", trx[cols], cond.train.y[cols], [q, qs], n, seed,
                                            T.steps_for(len(cols), epochs=hp["epochs"]), lr=hp["lr"])
            accs["normal"].append(float((lg.argmax(1) == cond.test.y).mean()))
            accs["swapped"].append(float((lgs.argmax(1) == cond.test.y).mean()))
        for name in accs:
            res.append(dict(method="cnn1d", queries=name, k=str(k), top1=float(np.mean(accs[name]))))
    return pd.DataFrame(res)


def motion_vs_gap(n):
    """Per-sign wrist path length (F4 location features, shoulder-width units) against the per-sign CNN-minus-DTW accuracy gap."""
    cond = X.build_condition(C.vocab(n), "F4", "zero")
    loc = [slice(273, 276), slice(549, 552)]
    motion = {}
    for v, x in zip(cond.train.videos, cond.train.seqs):
        tot = 0.0
        for sl in loc:
            p = x[:, sl]
            ok = np.linalg.norm(p, axis=1) > 0
            both = ok[1:] & ok[:-1]
            if both.any():
                tot += float(np.linalg.norm(np.diff(p, axis=0), axis=1)[both].sum())
        motion.setdefault(v.sign, []).append(tot)
    mot = pd.Series({s: np.mean(v) for s, v in motion.items()})
    d = pd.concat([pd.read_csv(C.RESULTS / "raw" / "test" / f"{m}.csv.gz").assign(m=m) for m in ("dtw1nn", "cnn1d")])
    d["k"] = d["k"].astype(str)
    d = d[(d.vocab == n) & (d.k == "full") & (d.fset == "F1")]
    acc = d.groupby(["m", "sign"]).apply(lambda g: (g["rank"] <= 1).mean(), include_groups=False).unstack(0)
    signs = C.vocab(n)
    out = pd.DataFrame({"sign": [signs[i] for i in acc.index], "motion": [mot[signs[i]] for i in acc.index],
                        "dtw": acc["dtw1nn"].values, "cnn": acc["cnn1d"].values})
    out["gap"] = out.cnn - out.dtw
    rho = spearmanr(out.motion, out.gap)
    rng = np.random.default_rng(0)
    boots = [spearmanr(*out.iloc[rng.integers(0, len(out), len(out))][["motion", "gap"]].values.T)[0] for _ in range(2000)]
    rho_dtw = spearmanr(out.motion, out.dtw)[0]
    return out, dict(rho=float(rho[0]), lo=float(np.nanquantile(boots, 0.025)), hi=float(np.nanquantile(boots, 0.975)),
                     rho_dtw_acc=float(rho_dtw), n=len(out))


def run(n=50):
    out = {}
    su = slot_usage(C.vocab(n))
    su["dtw_acc_full"] = su.signer.map(per_signer_acc("dtw1nn", n, "full"))
    su["cnn_acc_full"] = su.signer.map(per_signer_acc("cnn1d", n, "full"))
    su.to_csv(C.RESULTS / "tables" / "why_slot_usage.csv", index=False)
    out["slot_left_only_min"] = float(su.left_only.min())
    out["slot_left_only_max"] = float(su.left_only.max())
    out["slot_right_only_min"] = float(su.right_only.min())
    out["slot_right_only_max"] = float(su.right_only.max())
    out["slot_n_left_majority"] = int((su.left_only > su.right_only).sum())
    sw = slot_swap_test(n)
    sw.to_csv(C.RESULTS / "tables" / "why_slot_swap.csv", index=False)
    out["swap"] = {f"{r.method}_{r.queries}_{r.k}": r.top1 for r in sw.itertuples()}
    mt, mstat = motion_vs_gap(n)
    mt.to_csv(C.RESULTS / "tables" / "why_motion_gap.csv", index=False)
    out["motion"] = mstat
    (C.RESULTS / "why.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


if __name__ == "__main__":
    R.guard_test_split()
    print(json.dumps(run(), indent=1))
