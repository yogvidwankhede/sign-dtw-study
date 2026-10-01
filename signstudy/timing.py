"""Per-query latency and CPU load. Run as: NUMBA_NUM_THREADS=1 python -m signstudy.timing --threads 1 --vocab 50
Latency covers the classification step only (feature sequence -> label); landmark extraction is timed separately.
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import dtwlib as L
from . import experiment as X
from . import features as F
from . import legacy as LG
from . import run as R
from . import train as T


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measure(fn, n):
    lat = []
    c0, t0 = _cpu(), time.perf_counter()
    for i in range(n):
        s = time.perf_counter()
        fn(i)
        lat.append(time.perf_counter() - s)
    wall, cpu = time.perf_counter() - t0, _cpu() - c0
    lat = np.array(lat) * 1000
    return dict(median_ms=float(np.median(lat)), p95_ms=float(np.quantile(lat, 0.95)), cpu_per_wall=cpu / wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--radius", type=int, default=-1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cond = X.build_condition(C.vocab(a.vocab))
    nq = min(a.queries, len(cond.val.seqs))
    qs = [F.dtw_sequence(x) for x in cond.val.seqs[:nq]]
    qr = np.stack([F.resample(x) for x in cond.val.seqs[:nq]])
    rows = []
    for k in (10, "full"):
        order = X.orders_for(cond, 0)
        cols = np.array(sorted(i for s in cond.signs for i in (order[s] if k == "full" else order[s][:k])))
        refs = [F.dtw_sequence(cond.train.seqs[i]) for i in cols]
        rr = np.stack([F.resample(cond.train.seqs[i]) for i in cols])
        L.dtw_matrix(qs[:1], refs[:2], a.radius)  # compile / warm up
        rows.append(dict(method="dtw1nn", k=k, refs=len(cols), **measure(lambda i: L.dtw_matrix([qs[i]], refs, a.radius), nq)))
        rows.append(dict(method="euclid1nn", k=k, refs=len(cols), **measure(lambda i: L.euclid_matrix(qr[i:i + 1], rr), nq)))
        if k == 10:
            sub = list(range(min(len(cols), 500)))
            rows.append(dict(method="fastdtw_r1", k=k, refs=len(sub), **measure(
                lambda i: [LG.fastdtw_distance(cond.val.seqs[i], cond.train.seqs[cols[j]]) for j in sub], min(nq, 20))))
    dev_names = ["cpu"] + (["mps"] if torch.backends.mps.is_available() and a.threads > 1 else [])
    train_x = np.stack([F.resample(x) for x in cond.train.seqs[:500]])
    for m in ("cnn1d", "bigru", "transformer"):
        for dn in dev_names:
            from . import models as M
            net = M.build(m, train_x.shape[-1], a.vocab).to(dn).eval()
            xs = [torch.tensor(qr[i:i + 1], dtype=torch.float32, device=dn) for i in range(nq)]

            def fwd(i):
                with torch.no_grad():
                    net(xs[i]).cpu()
            fwd(0)
            rows.append(dict(method=f"{m}_{dn}", k="n/a", refs=0, **measure(fwd, nq)))
    df = pd.DataFrame(rows).assign(threads=a.threads, vocab=a.vocab)
    path = C.RESULTS / "timing" / f"latency_threads{a.threads}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    hw = dict(machine=platform.platform(), cpu=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
              capture_output=True, text=True).stdout.strip(), cores=os.cpu_count(), numba_threads=os.environ.get("NUMBA_NUM_THREADS"),
              torch_threads=a.threads, note="other processes were running on this machine during measurement")
    (path.parent / "hardware.json").write_text(json.dumps(hw, indent=1))
    print(df.to_string())


if __name__ == "__main__":
    main()
