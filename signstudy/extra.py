"""Part B / extra baselines (PREREG_ADDENDUM.md): kappa-NN voting, Fisher-weighted DTW, MiniRocket, longer training,
per-k tuning. Everything here is exploratory for PopSign and secondary for ASL Citizen."""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import experiment as X
from . import features as F
from . import run as R

KAPPAS = (1, 3, 5)


def fisher_weights(cond, cols, sd):
    """Per-dimension weights from the selected references: between-sign over within-sign variance of per-video mean features
    (features already divided by sd), normalised to mean 1 and clipped to [0.1, 10]. One reference per sign -> all ones."""
    y = cond.train.y[cols]
    means = np.stack([cond.train.seqs[i].mean(0) / sd for i in cols])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < 2 or len(classes) < 2:
        return np.ones(means.shape[1], dtype=np.float32)
    grand = means.mean(0)
    between = np.zeros(means.shape[1])
    within = np.zeros(means.shape[1])
    for c in classes:
        m = means[y == c]
        between += len(m) * (m.mean(0) - grand) ** 2
        within += ((m - m.mean(0)) ** 2).sum(0)
    w = (between / max(len(classes) - 1, 1)) / (within / max(len(y) - len(classes), 1) + 1e-9)
    w = np.clip(w / max(w.mean(), 1e-9), 0.1, 10.0)
    return w.astype(np.float32)


def eval_minirocket(cond, split_name, vocabs, ks, seeds, n_kernels=10000, label="minirocket"):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    split = getattr(cond, split_name)
    qx = np.stack([F.resample(x) for x in split.seqs])
    trx = np.stack([F.resample(x) for x in cond.train.seqs])
    out = []
    for n in vocabs:
        qmask = np.flatnonzero(split.y < n)
        for k in ks:
            for seed in (seeds if k != "full" else seeds[:1]):
                cols = R._ref_cols(cond, n, k, seed)
                flat = trx[cols].reshape(-1, trx.shape[-1])
                mu, sd = flat.mean(0), np.where(flat.std(0) < 1e-3, 1.0, flat.std(0))
                xtr = ((trx[cols] - mu) / sd).transpose(0, 2, 1).astype(np.float32)
                xq = ((qx[qmask] - mu) / sd).transpose(0, 2, 1).astype(np.float32)
                t0 = time.time()
                mr = MiniRocket(n_kernels=n_kernels, random_state=seed)
                ftr = mr.fit_transform(xtr)
                clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(ftr, cond.train.y[cols])
                sc = clf.decision_function(mr.transform(xq))
                if sc.ndim == 1:                      # two classes: sklearn returns one score per sample
                    sc = np.stack([-sc, sc], axis=1)
                sc = -sc
                meta = dict(method=label, split=split_name, vocab=n, k=k, seed=seed if k != "full" else -1,
                            fset=cond.fset, missing=cond.missing, dominant=cond.dominant)
                out.append(R._rows(split, sc.astype(np.float32), split.y[qmask], n, meta))
                print(label, n, k, seed, round(time.time() - t0, 1), "s", flush=True)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ stages
def stage_tune_kappa(n_tune):
    """kappa for the kNN variants, chosen on validation signers (n_tune signs, k=10, seed 0) with the family's tuned settings."""
    from . import pipeline as P
    t = P._tuned()["families"]
    out = {"n_tune": n_tune}
    for fam, fn in (("dtw", P._dtw_family), ("euclid", P._euclid_family)):
        cond = X.build_condition(C.vocab(n_tune), "F1", t[fam]["missing"])
        res = []
        for kap in KAPPAS:
            df = fn(cond, "val", t[fam], [n_tune], kappa=kap, ks=[10]).query("seed == 0")
            res.append(({"kappa": kap}, float((df["rank"] <= 1).mean())))
            print(fam, res[-1], flush=True)
        out[fam], _ = P._pick(res)
        out[fam + "_grid"] = res
    (C.RESULTS / "tuned_extra.json").write_text(json.dumps(out, indent=1))
    return out


def stage_extra(split, n_max, vocabs, which):
    from . import pipeline as P
    if split == "test":
        R.guard_test_split()
    t = P._tuned()["families"]
    ke = json.loads((C.RESULTS / "tuned_extra.json").read_text())
    signs = C.vocab(n_max)
    if "knn" in which:
        cond = X.build_condition(signs, "F1", t["dtw"]["missing"])
        R.save(P._dtw_family(cond, split, t["dtw"], vocabs, label="dtw_knn", kappa=ke["dtw"]["kappa"]), "extra_dtw_knn", split)
        cond = X.build_condition(signs, "F1", t["euclid"]["missing"])
        R.save(P._euclid_family(cond, split, t["euclid"], vocabs, label="euclid_knn", kappa=ke["euclid"]["kappa"]), "extra_euclid_knn", split)
    if "wdtw" in which:
        cond = X.build_condition(signs, "F1", t["dtw"]["missing"])
        cfg = dict(t["dtw"], scale="std")
        R.save(P._dtw_family(cond, split, cfg, vocabs, label="wdtw1nn", weights=fisher_weights), "extra_wdtw", split)
    if "minirocket" in which:
        cond = X.build_condition(signs, "F1", t["cnn1d"]["missing"])
        R.save(eval_minirocket(cond, split, [v for v in vocabs if v <= 100], list(C.REFS_PER_SIGN), list(C.SEEDS), n_kernels=2000),
               "extra_minirocket", split)
    if "long" in which:
        for m in P.MODELS:
            hp = t[m]
            cond = X.build_condition(signs, "F1", hp["missing"])
            df, _ = R.eval_learned(cond, split, m, vocabs, list(C.REFS_PER_SIGN), list(C.SEEDS), lr=hp["lr"], epochs=200,
                                   cap=12000, tag=m + "_long")
            R.save(df, f"extra_{m}_long", split)


PERK_KS = {1: [1, 2], 5: [5], 10: [10, 20], "full": ["full"]}


def stage_perk(split, n):
    """Per-k validation tuning (missing hands fixed to the tuned choice), then test evaluation with the k-specific settings."""
    from . import pipeline as P
    if split == "test":
        R.guard_test_split()
    t = P._tuned()["families"]
    signs = C.vocab(n)
    avail = set(C.REFS_PER_SIGN)
    tuned, rows = {}, []
    for kk, test_ks in PERK_KS.items():
        if kk not in avail:
            continue
        test_ks = [k for k in test_ks if k in avail]
        tuned[str(kk)] = {}
        # DTW and Euclidean
        cond = X.build_condition(signs, "F1", t["dtw"]["missing"])
        res = []
        for sc in P.SCALES:
            for r in P.RADII:
                df = P._dtw_family(cond, "val", dict(missing=t["dtw"]["missing"], scale=sc, radius=r), [n], ks=[kk]).query("seed <= 0 or k == 'full'")
                res.append((dict(scale=sc, radius=r), float((df["rank"] <= 1).mean())))
        cfg, _ = P._pick(res)
        tuned[str(kk)]["dtw"] = cfg
        rows.append(P._dtw_family(cond, split, dict(missing=t["dtw"]["missing"], **cfg), [n], label="dtw_perk", ks=test_ks))
        cond = X.build_condition(signs, "F1", t["euclid"]["missing"])
        res = []
        for sc in P.SCALES:
            df = P._euclid_family(cond, "val", dict(missing=t["euclid"]["missing"], scale=sc), [n], ks=[kk]).query("seed <= 0 or k == 'full'")
            res.append((dict(scale=sc), float((df["rank"] <= 1).mean())))
        cfg, _ = P._pick(res)
        tuned[str(kk)]["euclid"] = cfg
        rows.append(P._euclid_family(cond, split, dict(missing=t["euclid"]["missing"], **cfg), [n], label="euclid_perk", ks=test_ks))
        # learned models
        for m in P.MODELS:
            cond = X.build_condition(signs, "F1", t[m]["missing"])
            res = []
            for lr in P.LR_GRID:
                for ep in P.EPOCH_GRID:
                    df, _ = R.eval_learned(cond, "val", m, [n], [kk], [0], lr=lr, epochs=ep)
                    res.append((dict(lr=lr, epochs=ep), float((df["rank"] <= 1).mean())))
            cfg, _ = P._pick(res)
            tuned[str(kk)][m] = cfg
            df, _ = R.eval_learned(cond, split, m, [n], test_ks, list(C.SEEDS), lr=cfg["lr"], epochs=cfg["epochs"], tag=m + "_perk")
            rows.append(df)
        print("per-k tuned", kk, tuned[str(kk)], flush=True)
        R.save(pd.concat(rows, ignore_index=True), "extra_perk", split)
        (C.RESULTS / "tuned_perk.json").write_text(json.dumps(tuned, indent=1))
    return tuned


# ------------------------------------------------------- second batch (PREREG_ADDENDUM.md items 7-9)
def eval_embedding(cond, split_name, n_eval, pre_from, ks, seeds, lr=1e-3, epochs=20, label="embed_nn"):
    """Learned-embedding nearest neighbour: a CNN is trained on signs with index >= pre_from only; queries of the first n_eval signs
    are matched to k references per sign in the CNN's pooled feature space (L2-normalised, Euclidean)."""
    import torch
    from . import models as M
    from . import train as T
    split = getattr(cond, split_name)
    trx = np.stack([F.resample(x) for x in cond.train.seqs])
    qx = np.stack([F.resample(x) for x in split.seqs])
    pre = np.flatnonzero(cond.train.y >= pre_from)
    n_cls = int(cond.train.y.max()) + 1 - pre_from
    flat = trx[pre].reshape(-1, trx.shape[-1])
    mu, sd = flat.mean(0), np.where(flat.std(0) < 1e-3, 1.0, flat.std(0))
    out = []
    dev = T.device()
    for seed in seeds:
        steps = T.steps_for(len(pre), epochs=epochs)
        _, secs, model = T.fit_predict("cnn1d", trx[pre], cond.train.y[pre] - pre_from, [], n_cls, seed, steps, lr=lr)
        model.eval()

        def embed(x):
            xt = torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev)
            with torch.no_grad():
                h = torch.cat([model.net(xt[i:i + 512].transpose(1, 2)).mean(-1) for i in range(0, len(xt), 512)])
                h2 = torch.cat([model.net(xt[i:i + 512].transpose(1, 2)).amax(-1) for i in range(0, len(xt), 512)])
            e = torch.cat([h, h2], dim=-1).cpu().numpy()
            return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)

        etr, eq = embed(trx), embed(qx)
        qmask = np.flatnonzero(split.y < n_eval)
        d_all = np.sqrt(np.maximum(((eq[qmask][:, None, :] - etr[None, :, :]) ** 2).sum(-1), 0)).astype(np.float32)
        print(label, "seed", seed, "embedding trained on", n_cls, "other signs in", round(secs), "s", flush=True)
        for k in ks:
            if k == "full" and seed != seeds[0]:
                continue
            cols = R._ref_cols(cond, n_eval, k, seed)
            sc = X.sign_min_scores(d_all[:, cols], cond.train.y[cols], n_eval)
            meta = dict(method=label, split=split_name, vocab=n_eval, k=k, seed=seed if k != "full" else -1,
                        fset=cond.fset, missing=cond.missing, dominant=cond.dominant)
            out.append(R._rows(split, sc, split.y[qmask], n_eval, meta))
    return pd.concat(out, ignore_index=True)


def eval_softdtw_proto(cond, split_name, vocabs, ks, seeds, gamma, radius, standardise, label="softdtw_proto"):
    from tslearn.barycenters import softdtw_barycenter
    from . import dtwlib as L
    split = getattr(cond, split_name)
    n_all = max(vocabs)
    out = []
    for k in ks:
        for seed in (seeds if k != "full" else seeds[:1]):
            order = X.orders_for(cond, seed)
            sd = (R._frame_std([cond.train.seqs[i] for i in R._ref_cols(cond, n_all, k, seed)]) if standardise else np.float32(1.0))
            protos = []
            for s in cond.signs[:n_all]:
                idx = order[s] if k == "full" else order[s][:k]
                xs = np.stack([F.resample(cond.train.seqs[i]) / sd for i in idx])
                protos.append(xs[0] if len(xs) == 1 else
                              softdtw_barycenter(xs, gamma=gamma, max_iter=50, tol=1e-3).astype(np.float32))
            q = [F.dtw_sequence(x) / sd for x in split.seqs]
            dist = L.dtw_matrix(q, protos, radius)
            for n in vocabs:
                qmask = split.y < n
                meta = dict(method=label, split=split_name, vocab=n, k=k, seed=seed if k != "full" else -1,
                            fset=cond.fset, missing=cond.missing, dominant=cond.dominant, gamma=gamma)
                out.append(R._rows(split, dist[qmask][:, :n], split.y[qmask], n, meta))
    return pd.concat(out, ignore_index=True)


def stage_tune_gamma(n_tune=50):
    from . import pipeline as P
    t = P._tuned()["families"]["dtw"]
    cond = X.build_condition(C.vocab(n_tune), "F1", t["missing"])
    res = []
    for g in (0.1, 1.0):
        df = eval_softdtw_proto(cond, "val", [n_tune], [10], [0], g, t["radius"], t["scale"] == "std")
        res.append(({"gamma": g}, float((df["rank"] <= 1).mean())))
        print("softdtw gamma", res[-1], flush=True)
    best, _ = P._pick(res)
    ke = json.loads((C.RESULTS / "tuned_extra.json").read_text())
    ke["softdtw"] = best
    ke["softdtw_grid"] = res
    (C.RESULTS / "tuned_extra.json").write_text(json.dumps(ke, indent=1))
    return best


def stage_second_batch(split, n_max, which, pre_from):
    from . import pipeline as P
    if split == "test":
        R.guard_test_split()
    t = P._tuned()["families"]
    signs = C.vocab(n_max)
    if "embed" in which:
        hp = t["cnn1d"]
        cond = X.build_condition(signs, "F1", hp["missing"])
        R.save(eval_embedding(cond, split, 50, pre_from, list(C.REFS_PER_SIGN), list(C.SEEDS), lr=hp["lr"], epochs=hp["epochs"]),
               "extra_embed_nn", split)
    if "softdtw" in which:
        ke = json.loads((C.RESULTS / "tuned_extra.json").read_text())
        cond = X.build_condition(C.vocab(50), "F1", t["dtw"]["missing"])
        ks = [k for k in C.REFS_PER_SIGN if k != "full" and k <= 10]
        R.save(eval_softdtw_proto(cond, split, [10, 50], ks, list(C.SEEDS), ke["softdtw"]["gamma"], t["dtw"]["radius"],
                                  t["dtw"]["scale"] == "std"), "extra_softdtw_proto", split)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune_kappa", "run", "perk", "tune_gamma", "second"])
    ap.add_argument("--pre-from", type=int, default=100, help="embedding baseline: first sign index used for embedding training")
    ap.add_argument("--split", default="val")
    ap.add_argument("--vocab", type=int, default=50, help="largest vocabulary / tuning vocabulary")
    ap.add_argument("--vocabs", type=int, nargs="*")
    ap.add_argument("--which", nargs="*", default=["knn", "wdtw", "minirocket", "long"])
    a = ap.parse_args()
    if a.stage == "tune_kappa":
        stage_tune_kappa(a.vocab)
    elif a.stage == "tune_gamma":
        stage_tune_gamma(a.vocab)
    elif a.stage == "second":
        stage_second_batch(a.split, a.vocab, a.which, a.pre_from)
    elif a.stage == "run":
        stage_extra(a.split, a.vocab, a.vocabs or [a.vocab], a.which)
    else:
        stage_perk(a.split, a.vocab)
