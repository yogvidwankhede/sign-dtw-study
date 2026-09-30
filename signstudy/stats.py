"""Signer-clustered bootstrap. Seeds are averaged first (per signer), then signers are resampled with replacement."""
import numpy as np
import pandas as pd

KEYS = ["method", "vocab", "k", "fset", "missing", "dominant"]
B_DEFAULT = 10000
BOOT_SEED = 12345


def signer_table(df, top=1):
    """One row per (condition, signer): seed-averaged number of correct queries `c` and query count `n`."""
    d = df.assign(ok=(df["rank"] <= top).astype(float))
    per_seed = d.groupby(KEYS + ["seed", "signer"], dropna=False).agg(c=("ok", "sum"), n=("ok", "size")).reset_index()
    return per_seed.groupby(KEYS + ["signer"], dropna=False).agg(c=("c", "mean"), n=("n", "first")).reset_index()


def _resample_index(n_signers, B, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n_signers, size=(B, n_signers))


def accuracy_ci(tab, conf=0.95, B=B_DEFAULT):
    """Point estimate and percentile interval per condition."""
    rows = []
    for key, g in tab.groupby(KEYS, dropna=False):
        g = g.sort_values("signer")
        c, n = g["c"].to_numpy(), g["n"].to_numpy()
        idx = _resample_index(len(g), B)
        acc = c[idx].sum(1) / n[idx].sum(1)
        lo, hi = np.quantile(acc, [(1 - conf) / 2, 1 - (1 - conf) / 2])
        rows.append(dict(zip(KEYS, key), acc=c.sum() / n.sum(), lo=lo, hi=hi, signers=len(g), queries=int(n.sum())))
    return pd.DataFrame(rows)


def paired_difference(tab, a, b, where, conf=0.95, B=B_DEFAULT):
    """acc(method a) - acc(method b) on the same signers and resamples. `where` fixes vocab/k/fset/... (dict)."""
    def pick(m):
        t = tab[tab["method"] == m]
        for col, val in where.items():
            t = t[t[col] == val]
        return t.sort_values("signer")
    ta, tb = pick(a), pick(b)
    if len(ta) == 0 or len(tb) == 0 or list(ta["signer"]) != list(tb["signer"]):
        return None
    idx = _resample_index(len(ta), B)
    da = ta["c"].to_numpy()[idx].sum(1) / ta["n"].to_numpy()[idx].sum(1)
    db = tb["c"].to_numpy()[idx].sum(1) / tb["n"].to_numpy()[idx].sum(1)
    diff = da - db
    lo, hi = np.quantile(diff, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    point = ta["c"].sum() / ta["n"].sum() - tb["c"].sum() / tb["n"].sum()
    return dict(a=a, b=b, diff=point, lo=lo, hi=hi, conf=conf, signers=len(ta), **where)
