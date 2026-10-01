"""Recompute some test-split rows from the landmark cache and compare with the stored results (exact equality of ranks).
Usage: python scripts/check_rerun.py   (needs data/landmarks; refuses unless the pre-registration guard passes)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signstudy import config as C  # noqa: E402
from signstudy import experiment as X  # noqa: E402
from signstudy import pipeline as P  # noqa: E402
from signstudy import run as R  # noqa: E402


def compare(new, name, **flt):
    old = pd.read_csv(C.RESULTS / "raw" / "test" / f"{name}.csv.gz")
    old["k"] = old["k"].astype(str)
    new = new.assign(k=new["k"].astype(str))
    for col, val in flt.items():
        old, new = old[old[col] == val], new[new[col] == val]
    key = ["vocab", "k", "seed", "signer", "sign"]
    # several queries share (signer, sign); rows are written in query order, so the position within the group identifies them
    old = old.assign(qid=old.groupby(key).cumcount())
    new = new.assign(qid=new.groupby(key).cumcount())
    m = old.merge(new, on=key + ["qid"], suffixes=("_old", "_new"), validate="one_to_one")
    same_rank = (m["rank_old"] == m["rank_new"]).mean()
    same_pred = (m["pred_old"] == m["pred_new"]).mean()
    print(f"{name} {flt}: {len(m)} matched rows | identical ranks {same_rank:.4f} | identical predictions {same_pred:.4f}")
    return same_rank


def main():
    R.guard_test_split()
    t = P._tuned()["families"]
    # Queries are drawn once for the largest vocabulary (50) and restricted for smaller ones, exactly as in pipeline.stage_main.
    signs = C.vocab(50)
    cond = X.build_condition(signs, "F1", t["dtw"]["missing"])
    compare(P._dtw_family(cond, "test", t["dtw"], [10]), "dtw1nn", vocab=10)
    cond = X.build_condition(signs, "F1", t["euclid"]["missing"])
    compare(P._euclid_family(cond, "test", t["euclid"], [10]), "euclid1nn", vocab=10)
    hp = t["cnn1d"]
    cond = X.build_condition(signs, "F1", hp["missing"])
    df, _ = R.eval_learned(cond, "test", "cnn1d", [10], [5], [0], lr=hp["lr"], epochs=hp["epochs"])
    compare(df, "cnn1d", vocab=10, k="5", seed=0)


if __name__ == "__main__":
    main()
