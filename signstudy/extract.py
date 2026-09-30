"""Download PopSign tars one (sign, split) at a time, extract MediaPipe Holistic landmarks, delete the videos.

Output: data/landmarks/<sign>/<split>.npz (float16 landmarks, NaN where a part was not detected).
Resumable: an existing .npz is skipped. Videos are never kept or redistributed.
"""
import json
import multiprocessing as mp
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config as C

_HOLISTIC = None
_FACE_IDX = None


def face_indices():
    import mediapipe as mpipe
    idx = sorted({i for pair in mpipe.solutions.face_mesh.FACEMESH_CONTOURS for i in pair})
    return idx


def _init_worker():
    global _HOLISTIC, _FACE_IDX
    import cv2
    import mediapipe as mpipe
    cv2.setNumThreads(1)
    _FACE_IDX = np.array(face_indices())
    _HOLISTIC = mpipe.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _arr(lm, n, dims):
    if lm is None:
        return np.full((n, dims), np.nan, dtype=np.float32)
    get = (lambda p: (p.x, p.y, p.z, p.visibility)) if dims == 4 else (lambda p: (p.x, p.y, p.z))
    return np.array([get(p) for p in lm.landmark], dtype=np.float32)


def process_video(path):
    """Return dict of per-frame landmark arrays for one video, plus frame size used for aspect correction."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    left, right, pose, face = [], [], [], []
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        scale = C.SHORT_SIDE / min(fh, fw)
        frame = cv2.resize(frame, (round(fw * scale), round(fh * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        res = _HOLISTIC.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        left.append(_arr(res.left_hand_landmarks, 21, 3))
        right.append(_arr(res.right_hand_landmarks, 21, 3))
        pose.append(_arr(res.pose_landmarks, 33, 4))
        if res.face_landmarks is None:
            face.append(np.full((len(_FACE_IDX), 3), np.nan, dtype=np.float32))
        else:
            face.append(_arr(res.face_landmarks, 468, 3)[_FACE_IDX])
    cap.release()
    if not left:
        return None
    return dict(left=np.stack(left), right=np.stack(right), pose=np.stack(pose), face=np.stack(face), size=(w, h))


def download(url, dest, retries=5):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "3", "-o", str(dest), url])
        if r.returncode == 0:
            return
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def pack(results, names, out):
    lens = np.array([r["left"].shape[0] for r in results])
    offsets = np.concatenate([[0], np.cumsum(lens)])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, offsets=offsets, names=np.array(names),
        sizes=np.array([r["size"] for r in results]),
        left=np.concatenate([r["left"] for r in results]).astype(np.float16),
        right=np.concatenate([r["right"] for r in results]).astype(np.float16),
        pose=np.concatenate([r["pose"] for r in results]).astype(np.float16),
        face=np.concatenate([r["face"] for r in results]).astype(np.float16))
    tmp.replace(out)


def versions():
    import cv2
    import mediapipe as mpipe
    return dict(mediapipe=mpipe.__version__, opencv=cv2.__version__, numpy=np.__version__,
                python=sys.version.split()[0], machine=platform.platform(), processor=platform.processor())


def run(signs, workers=8, log=C.LANDMARKS / "EXTRACTION_LOG.jsonl"):
    jobs = [(s, sp) for s in signs for sp in C.SPLITS if not (C.LANDMARKS / s / f"{sp}.npz").exists()]
    if not jobs:
        print("nothing to do")
        return
    tmp_root = C.DATA / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    def fetch(job):
        sign, split = job
        tar_path = tmp_root / f"{sign}_{split}.tar"
        t = time.time()
        download(f"{C.BASE_URL}/{C.DATASET}/{C.CATEGORY}/{split}/{sign}.tar", tar_path)
        return job, tar_path, time.time() - t

    ctx = mp.get_context("spawn")
    with ThreadPoolExecutor(max_workers=2) as dl, ctx.Pool(workers, initializer=_init_worker) as pool:
        futures = [dl.submit(fetch, j) for j in jobs[:2]]
        nxt = 2
        for _ in range(len(jobs)):
            (sign, split), tar_path, t_dl = futures.pop(0).result()
            if nxt < len(jobs):
                futures.append(dl.submit(fetch, jobs[nxt]))
                nxt += 1
            vid_dir = tmp_root / f"{sign}_{split}"
            shutil.rmtree(vid_dir, ignore_errors=True)
            with tarfile.open(tar_path) as tf:
                tf.extractall(vid_dir)
            tar_path.unlink()
            paths = sorted(vid_dir.rglob("*.mp4"))
            t = time.time()
            results = pool.map(process_video, paths, chunksize=2)
            keep = [(p, r) for p, r in zip(paths, results) if r is not None]
            pack([r for _, r in keep], [p.name for p, _ in keep], C.LANDMARKS / sign / f"{split}.npz")
            shutil.rmtree(vid_dir, ignore_errors=True)
            rec = dict(sign=sign, split=split, videos=len(paths), kept=len(keep), download_s=round(t_dl, 1),
                       extract_s=round(time.time() - t, 1), **versions())
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec["sign"], rec["split"], rec["kept"], "videos", rec["extract_s"], "s", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1])
    run(C.vocab(n), workers=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
