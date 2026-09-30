"""Load extracted landmarks, enforce signer-disjoint splits, and draw the nested subsets used by every method."""
import zlib

import numpy as np

from . import config as C


class Video:
    """One video's landmarks (float32) plus labels. Arrays have shape (frames, points, dims)."""
    __slots__ = ("sign", "split", "signer", "name", "left", "right", "pose", "face", "size")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def signer_of(name):
    return name.split("-")[0]


def load_sign(sign, split, with_face=False):
    path = C.LANDMARKS / sign / f"{split}.npz"
    z = np.load(path, allow_pickle=False)
    off, names = z["offsets"], z["names"]
    left, right, pose = z["left"], z["right"], z["pose"]
    face = z["face"] if with_face else None
    out = []
    for i, nm in enumerate(names):
        a, b = off[i], off[i + 1]
        out.append(Video(sign=sign, split=split, signer=signer_of(str(nm)), name=str(nm),
                         left=left[a:b].astype(np.float32), right=right[a:b].astype(np.float32),
                         pose=pose[a:b].astype(np.float32),
                         face=None if face is None else face[a:b].astype(np.float32),
                         size=tuple(z["sizes"][i])))
    return out


def load_split(signs, split, with_face=False):
    vids = []
    for s in signs:
        vids.extend(load_sign(s, split, with_face))
    return vids


def check_signer_disjoint(signs):
    seen = {}
    for sp in C.SPLITS:
        ids = {v.signer for v in load_split(signs, sp)}
        for other, other_ids in seen.items():
            if ids & other_ids:
                raise AssertionError(f"signer overlap between {sp} and {other}: {sorted(ids & other_ids)[:5]}")
        seen[sp] = ids
    return {k: len(v) for k, v in seen.items()}


def draw_queries(videos, per_signer_sign=C.TEST_PER_SIGNER_SIGN, seed=C.MASTER_SEED):
    """Up to `per_signer_sign` videos per (signer, sign), same draw for every method."""
    groups = {}
    for i, v in enumerate(videos):
        groups.setdefault((v.signer, v.sign), []).append(i)
    rng = np.random.default_rng(seed)
    keep = []
    for key in sorted(groups):
        idx = sorted(groups[key], key=lambda i: videos[i].name)
        rng.shuffle(idx)
        keep.extend(idx[:per_signer_sign])
    return sorted(keep)


def reference_order(train_videos, signs, seed):
    """For each sign, train indices in a random (seeded) order; taking the first k gives nested subsets."""
    by_sign = {s: [] for s in signs}
    for i, v in enumerate(train_videos):
        by_sign[v.sign].append(i)
    order = {}
    for s in signs:
        idx = sorted(by_sign[s], key=lambda i: train_videos[i].name)
        rng = np.random.default_rng([seed, zlib.crc32(s.encode())])
        rng.shuffle(idx)
        order[s] = idx
    return order


def select_refs(order, k):
    """Indices of the first k references per sign ('full' = all)."""
    out = []
    for s, idx in order.items():
        out.extend(idx if k == "full" else idx[:k])
    return out
