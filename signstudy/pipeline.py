"""Stages: tune (validation only) -> main (test) -> ablate (test) -> legacy (test). See PREREG.md."""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import dataset as D
from . import experiment as X
from . import run as R

KS = list(C.REFS_PER_SIGN)
SEEDS = list(C.SEEDS)
RADII = (3, 5, 8, -1)
LR_GRID = (1e-3, 3e-4)
EPOCH_GRID = (20, 80)
MODELS = ("cnn1d", "bigru", "transformer")
TUNED = C.RESULTS / "tuned.json"


def available_vocab():
    n = 0
    for s in C.sign_list():
        if all((C.LANDMARKS / s / f"{sp}.npz").exists() for sp in C.SPLITS):
            n += 1
        else:
            break
    return n


def vocabs_up_to(n_avail):
    return sorted({v for v in C.VOCAB_SIZES if v <= n_avail} | {n_avail})


MISSING = ("zero", "ffill", "drop")
SCALES = ("raw", "std")
FAMILIES = ("dtw", "euclid") + MODELS


def _conds(signs):
    return {m: X.build_condition(signs, "F1", m) for m in MISSING}


def _pick(results):
    """results: list of (config dict, val top-1) in preference order; the first best one wins (ties -> earlier)."""
    best = max(range(len(results)), key=lambda i: (results[i][1], -i))
    return results[best][0], [(c, a) for c, a in results]


def _top1(df):
    return float(R.accuracy(df)["top1"].iloc[0])


def stage_tune(n_tune):
    """Choose every family's preprocessing and hyperparameters on VALIDATION signers only (vocab n_tune, k=10, seed 0)."""
    signs = C.vocab(n_tune)
    conds = _conds(signs)
    out = {"n_tune": n_tune, "k": 10, "seed": 0, "families": {}, "grids": {}}

    def run_dtw(cond, scale, r):
        if scale == "raw":
            return R.eval_distance(cond, "val", R.dtw_dist(cond, "val", r), "x", [n_tune], [10], [0])
        return R.eval_scaled(cond, "val", "x", [n_tune], [10], [0], kind="dtw", radius=r)

    res = []
    for m in MISSING:
        for sc in SCALES:
            for r in RADII:
                res.append((dict(missing=m, scale=sc, radius=r), _top1(run_dtw(conds[m], sc, r))))
                print("dtw", res[-1], flush=True)
    out["families"]["dtw"], out["grids"]["dtw"] = _pick(res)

    res = []
    for m in MISSING:
        for sc in SCALES:
            if sc == "raw":
                df = R.eval_distance(conds[m], "val", R.euclid_dist(conds[m], "val"), "x", [n_tune], [10], [0])
            else:
                df = R.eval_scaled(conds[m], "val", "x", [n_tune], [10], [0], kind="euclid")
            res.append((dict(missing=m, scale=sc), _top1(df)))
            print("euclid", res[-1], flush=True)
    out["families"]["euclid"], out["grids"]["euclid"] = _pick(res)

    for mod in MODELS:
        res = []
        for m in MISSING:
            for lr in LR_GRID:
                for ep in EPOCH_GRID:
                    df, _ = R.eval_learned(conds[m], "val", mod, [n_tune], [10], [0], lr=lr, epochs=ep)
                    res.append((dict(missing=m, lr=lr, epochs=ep), _top1(df)))
                    print(mod, res[-1], flush=True)
        out["families"][mod], out["grids"][mod] = _pick(res)
    out["provenance"] = R.provenance() if R.git("tag", "--list", C.PREREG_TAG) else {}
    TUNED.parent.mkdir(exist_ok=True)
    TUNED.write_text(json.dumps(out, indent=1))
    return out


def _tuned():
    return json.loads(TUNED.read_text())


def _dtw_family(cond, split, cfg, vocabs, label="dtw1nn"):
    if cfg["scale"] == "raw":
        d = R.dtw_dist(cond, split, cfg["radius"])
        return R.eval_distance(cond, split, d, label, vocabs, KS, SEEDS, extra=dict(radius=cfg["radius"], scale="raw"))
    return R.eval_scaled(cond, split, label, vocabs, KS, SEEDS, kind="dtw", radius=cfg["radius"]).assign(scale="std")


def _euclid_family(cond, split, cfg, vocabs, label="euclid1nn"):
    if cfg["scale"] == "raw":
        return R.eval_distance(cond, split, R.euclid_dist(cond, split), label, vocabs, KS, SEEDS, extra=dict(scale="raw"))
    return R.eval_scaled(cond, split, label, vocabs, KS, SEEDS, kind="euclid").assign(scale="std")


def stage_main(split, n_max, which):
    if split == "test":
        R.guard_test_split()
    t = _tuned()["families"]
    signs = C.vocab(n_max)
    vocabs = vocabs_up_to(n_max)
    outs = []
    if "dtw" in which:
        cond = X.build_condition(signs, "F1", t["dtw"]["missing"])
        outs.append(_dtw_family(cond, split, t["dtw"], vocabs))
        R.save(outs[-1], "dtw1nn", split)
    if "euclid" in which:
        cond = X.build_condition(signs, "F1", t["euclid"]["missing"])
        outs.append(_euclid_family(cond, split, t["euclid"], vocabs))
        R.save(outs[-1], "euclid1nn", split)
    if "dba" in which:
        cond = X.build_condition(signs, "F1", t["dtw"]["missing"])
        outs.append(R.eval_prototypes(cond, split, vocabs, KS, SEEDS, radius=t["dtw"]["radius"],
                                      standardise=t["dtw"]["scale"] == "std"))
        R.save(outs[-1], "dba_proto", split)
    for m in MODELS:
        if m not in which:
            continue
        hp = t[m]
        cond = X.build_condition(signs, "F1", hp["missing"])
        df, tm = R.eval_learned(cond, split, m, vocabs, KS, SEEDS, lr=hp["lr"], epochs=hp["epochs"])
        outs.append(df)
        R.save(df, m, split)
        R.save(tm, m + "_training_time", split)
    return outs


def stage_aug(split, n_aug):
    """Learned models with light augmentation (time stretch, noise, scale) at one vocabulary, same tuned settings."""
    if split == "test":
        R.guard_test_split()
    t = _tuned()["families"]
    rows = []
    for m in MODELS:
        hp = t[m]
        cond = X.build_condition(C.vocab(n_aug), "F1", hp["missing"])
        df, _ = R.eval_learned(cond, split, m, [n_aug], KS, SEEDS, lr=hp["lr"], epochs=hp["epochs"], aug=True,
                               tag=m + "_aug")
        rows.append(df)
        R.save(pd.concat(rows, ignore_index=True), "augmentation", split)
    return pd.concat(rows, ignore_index=True)


def stage_groups(n):
    """One- vs two-handed sign groups from TRAIN videos only (PREREG.md section 3): a sign is two-handed if both hands
    are detected in at least 50% of its hand-present frames."""
    from . import dataset as D
    rows = []
    for s in C.vocab(n):
        both = any_ = 0
        for v in D.load_sign(s, "train"):
            l, r = ~np.isnan(v.left[:, 0, 0]), ~np.isnan(v.right[:, 0, 0])
            both += int((l & r).sum())
            any_ += int((l | r).sum())
        rows.append(dict(sign=s, both_fraction=both / max(any_, 1), group="two-handed" if both / max(any_, 1) >= 0.5 else "one-handed"))
    out = C.RESULTS / "sign_groups.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def stage_ablate_drop(split, n_abl):
    """EXPLORATORY SUPPLEMENT (added after the freeze, see PREREG.md Deviations): richer feature sets with the tuned
    drop-frames setting, so that DTW's best case is not understated by zero-filling in the main ablation."""
    if split == "test":
        R.guard_test_split()
    t = _tuned()["families"]
    signs = C.vocab(n_abl)
    ks = [5, "full"]
    global KS
    saved, KS = KS, ks
    rows = []
    try:
        for fset, missing, dom in [("F3", "drop", False), ("F4", "drop", False), ("F5", "drop", False), ("F4", "drop", True)]:
            cond = X.build_condition(signs, fset, missing, dom)
            rows.append(_dtw_family(cond, split, t["dtw"], [n_abl]))
            rows.append(_euclid_family(cond, split, t["euclid"], [n_abl]))
            for m in MODELS:
                hp = t[m]
                df, _ = R.eval_learned(cond, split, m, [n_abl], ks, SEEDS, lr=hp["lr"], epochs=hp["epochs"])
                rows.append(df)
            R.save(pd.concat(rows, ignore_index=True), "ablation_drop", split)
    finally:
        KS = saved
    return pd.concat(rows, ignore_index=True)


def stage_ablate(split, n_abl):
    """Feature sets, missing-hand handling and handedness at vocabulary n_abl, k in {5, full}; each family keeps its
    tuned scaling / band / learning rate, only the factor under study changes."""
    if split == "test":
        R.guard_test_split()
    t = _tuned()["families"]
    signs = C.vocab(n_abl)
    rows = []
    settings = [(f, "zero", False) for f in ("F1", "F2", "F3", "F4", "F5", "F6")]
    settings += [("F1", "ffill", False), ("F1", "drop", False), ("F1", "zero", True)]
    ks = [5, "full"]
    global KS
    saved, KS = KS, ks
    try:
        for fset, missing, dom in settings:
            cond = X.build_condition(signs, fset, missing, dom)
            rows.append(_dtw_family(cond, split, t["dtw"], [n_abl]))
            rows.append(_euclid_family(cond, split, t["euclid"], [n_abl]))
            for m in MODELS:
                hp = t[m]
                df, _ = R.eval_learned(cond, split, m, [n_abl], ks, SEEDS, lr=hp["lr"], epochs=hp["epochs"])
                rows.append(df)
            R.save(pd.concat(rows, ignore_index=True), "ablation", split)
    finally:
        KS = saved
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "main", "ablate", "ablate_drop", "aug", "groups"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--vocab", type=int, default=10, help="number of signs to use (must be extracted)")
    ap.add_argument("--methods", nargs="*", default=["dtw", "euclid", "dba", *MODELS])
    a = ap.parse_args()
    if a.vocab > available_vocab():
        raise SystemExit(f"only {available_vocab()} signs are fully extracted")
    t0 = time.time()
    if a.stage == "tune":
        stage_tune(a.vocab)
    elif a.stage == "main":
        stage_main(a.split, a.vocab, a.methods)
    elif a.stage == "ablate_drop":
        stage_ablate_drop(a.split, a.vocab)
    elif a.stage == "aug":
        stage_aug(a.split, a.vocab)
    elif a.stage == "groups":
        stage_groups(a.vocab)
    else:
        stage_ablate(a.split, a.vocab)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
