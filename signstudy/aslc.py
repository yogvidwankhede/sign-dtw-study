"""ASL Citizen replication data (PREREG_ADDENDUM.md). Terms: Microsoft Research License (non-commercial research; no
distribution of the data or modifications; consistent with participants' consent; destroy personal data when the research
ends). So: only hand and pose landmarks are extracted (no face), videos are deleted right after extraction, landmarks stay in
the git-ignored data/ directory, and only aggregate results are committed.

    python -m signstudy.aslc prepare        # downloads the split CSVs, fixes the sign list and order
    SIGNSTUDY_DATASET=aslc python -m signstudy.aslc extract 300 [workers]
"""
import json
import multiprocessing as mp
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
MIN_TRAIN, MIN_VAL, MIN_TEST = 9, 2, 5


def _paths():
    from . import config as C
    return C.DATA / "aslc", C.DATA / "tmp_aslc"


def prepare():
    """Fetch split CSVs and fix the eligible signs (>= 9 train, >= 2 val, >= 5 test videos). Keys g0000.. are used for
    directory names so that no gloss text is needed outside data/aslc."""
    from remotezip import RemoteZip
    d, _ = _paths()
    d.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        for n in ("train", "val", "test"):
            (d / f"{n}.csv").write_bytes(z.read(f"ASL_Citizen/splits/{n}.csv"))
    sp = {n: pd.read_csv(d / f"{n}.csv") for n in ("train", "val", "test")}
    cnt = {n: sp[n].groupby("Gloss").size() for n in sp}
    ok = [g for g in sorted(cnt["train"].index)
          if cnt["train"].get(g, 0) >= MIN_TRAIN and cnt["val"].get(g, 0) >= MIN_VAL and cnt["test"].get(g, 0) >= MIN_TEST]
    keys = [f"g{i:04d}" for i in range(len(ok))]
    (d / "signs.json").write_text(json.dumps({"keys": keys, "glosses": ok}))
    print(f"{len(ok)} eligible signs of {sp['train'].Gloss.nunique()}; "
          f"participants train/val/test: {[sp[n]['Participant ID'].nunique() for n in sp]}")


def _fetch_all(rows, tmp, workers=6):
    """rows: list of (index, video file). Downloads each member by HTTP range request into tmp/<index>.mp4."""
    from remotezip import RemoteZip
    local = threading.local()

    def get(item):
        i, vf = item
        if not hasattr(local, "z"):
            local.z = RemoteZip(URL)
        for attempt in range(4):
            try:
                (tmp / f"{i}.mp4").write_bytes(local.z.read(f"ASL_Citizen/videos/{vf}"))
                return
            except Exception:
                time.sleep(3 * (attempt + 1))
                local.z = RemoteZip(URL)
        raise RuntimeError(f"could not fetch {vf}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, rows))


def _proc(path):
    from . import extract as E
    r = E.process_video(path)
    if r is not None:
        r["face"] = np.zeros((r["left"].shape[0], 0, 3), dtype=np.float32)   # face landmarks are never kept for ASL Citizen
    return r


def extract(n, workers=6, chunk_signs=25):
    from . import config as C
    from . import extract as E
    assert C.ASLC, "set SIGNSTUDY_DATASET=aslc"
    d, tmp = _paths()
    meta = json.loads((d / "signs.json").read_text())
    gloss_of = dict(zip(meta["keys"], meta["glosses"]))
    keys = C.vocab(n)
    sp = {s: pd.read_csv(d / f"{s}.csv") for s in C.SPLITS}
    log = C.LANDMARKS / "EXTRACTION_LOG.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=E._init_worker) as pool:
        for c0 in range(0, len(keys), chunk_signs):
            chunk = keys[c0:c0 + chunk_signs]
            todo = [(k, s) for k in chunk for s in C.SPLITS if not (C.LANDMARKS / k / f"{s}.npz").exists()]
            if not todo:
                continue
            glosses = {gloss_of[k]: k for k, _ in todo}
            items = []   # (index, key, split, participant, video file)
            for s in C.SPLITS:
                df = sp[s][sp[s].Gloss.isin([gloss_of[k] for k, sp_ in todo if sp_ == s])]
                for r in df.itertuples(index=False):
                    items.append((len(items), glosses[r.Gloss], s, r[0], r[1]))
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            t0 = time.time()
            _fetch_all([(i, vf) for i, _, _, _, vf in items], tmp)
            t_dl = time.time() - t0
            t0 = time.time()
            res = pool.map(_proc, [tmp / f"{i}.mp4" for i, *_ in items], chunksize=4)
            by = {}
            for it, r in zip(items, res):
                if r is not None:
                    by.setdefault((it[1], it[2]), []).append((f"{it[3]}-{gloss_of[it[1]]}-{it[4]}", r))
            for (k, s), lst in by.items():
                E.pack([r for _, r in lst], [nm for nm, _ in lst], C.LANDMARKS / k / f"{s}.npz")
            shutil.rmtree(tmp, ignore_errors=True)
            with open(log, "a") as f:
                f.write(json.dumps(dict(signs=chunk, videos=len(items), kept=sum(len(v) for v in by.values()),
                                        download_s=round(t_dl, 1), extract_s=round(time.time() - t0, 1), **E.versions())) + "\n")
            print(f"signs {c0}-{c0 + len(chunk) - 1}: {len(items)} videos, download {t_dl:.0f}s, extract {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    else:
        extract(int(sys.argv[2]), workers=int(sys.argv[3]) if len(sys.argv) > 3 else 6)
