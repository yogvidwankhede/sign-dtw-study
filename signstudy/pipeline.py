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
EPOCH_GRID = (15, 30)
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
    return [v for v in C.VOCAB_SIZES if v <= n_avail]


def stage_tune(n_tune):
    """Choose DTW radius and learned-model hyperparameters on VALIDATION signers only."""
    cond = X.build_condition(C.vocab(n_tune))
    out = {"n_tune": n_tune, "k": 10, "seed": 0, "radius": {}, "models": {}}
    for r in RADII:
        d = R.dtw_dist(cond, "val", r)
        acc = R.accuracy(R.eval_distance(cond, "val", d, "dtw", [n_tune], [10], [0]))["top1"].iloc[0]
        out["radius"][str(r)] = float(acc)
        print("radius", r, "val top1", round(acc, 4), flush=True)
    best = max(RADII, key=lambda r: (out["radius"][str(r)], -(r if r > 0 else 10**6)))
    out["chosen_radius"] = best
    for m in MODELS:
        scores = {}
        for lr in LR_GRID:
            for ep in EPOCH_GRID:
                df, _ = R.eval_learned(cond, "val", m, [n_tune], [10], [0], lr=lr, epochs=ep)
                scores[f"lr={lr},epochs={ep}"] = (float(R.accuracy(df)["top1"].iloc[0]), lr, ep)
        key = max(scores, key=lambda s: scores[s][0])
        out["models"][m] = {"chosen": key, "lr": scores[key][1], "epochs": scores[key][2],
                            "grid": {k: v[0] for k, v in scores.items()}}
        print(m, "chose", key, round(scores[key][0], 4), flush=True)
    out["provenance"] = R.provenance() if R.git("tag", "--list", "prereg-v1") else {}
    TUNED.parent.mkdir(exist_ok=True)
    TUNED.write_text(json.dumps(out, indent=1))
    return out


def _tuned():
    return json.loads(TUNED.read_text())


def stage_main(split, n_max, which):
    if split == "test":
        R.guard_test_split()
    t = _tuned()
    cond = X.build_condition(C.vocab(n_max))
    vocabs = vocabs_up_to(n_max)
    radius = t["chosen_radius"]
    outs = []
    if "dtw" in which:
        d = R.dtw_dist(cond, split, radius)
        outs.append(R.eval_distance(cond, split, d, "dtw1nn", vocabs, KS, SEEDS, extra=dict(radius=radius)))
        R.save(outs[-1], "dtw1nn", split)
    if "euclid" in which:
        d = R.euclid_dist(cond, split)
        outs.append(R.eval_distance(cond, split, d, "euclid1nn", vocabs, KS, SEEDS))
        R.save(outs[-1], "euclid1nn", split)
    if "dba" in which:
        outs.append(R.eval_prototypes(cond, split, vocabs, KS, SEEDS, radius=radius))
        R.save(outs[-1], "dba_proto", split)
    for m in MODELS:
        if m not in which:
            continue
        hp = t["models"][m]
        df, tm = R.eval_learned(cond, split, m, vocabs, KS, SEEDS, lr=hp["lr"], epochs=hp["epochs"])
        outs.append(df)
        R.save(df, m, split)
        R.save(tm, m + "_training_time", split)
    return outs


def stage_ablate(split, n_abl):
    if split == "test":
        R.guard_test_split()
    t = _tuned()
    signs = C.vocab(n_abl)
    ks = [5, "full"]
    rows = []
    settings = [(f, "zero", False) for f in ("F1", "F2", "F3", "F4", "F5", "F6")]
    settings += [("F1", "ffill", False), ("F1", "drop", False), ("F1", "zero", True)]
    for fset, missing, dom in settings:
        cond = X.build_condition(signs, fset, missing, dom)
        d = R.dtw_dist(cond, split, t["chosen_radius"])
        rows.append(R.eval_distance(cond, split, d, "dtw1nn", [n_abl], ks, SEEDS, extra=dict(radius=t["chosen_radius"])))
        rows.append(R.eval_distance(cond, split, R.euclid_dist(cond, split), "euclid1nn", [n_abl], ks, SEEDS))
        for m in MODELS:
            hp = t["models"][m]
            df, _ = R.eval_learned(cond, split, m, [n_abl], ks, SEEDS, lr=hp["lr"], epochs=hp["epochs"])
            rows.append(df)
        R.save(pd.concat(rows, ignore_index=True), "ablation", split)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "main", "ablate"])
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
    else:
        stage_ablate(a.split, a.vocab)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
