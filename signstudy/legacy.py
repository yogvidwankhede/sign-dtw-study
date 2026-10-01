"""The original repository's recognition logic, kept as a reference point (PREREG.md method 3) and for FastDTW timing.

Legacy = per-hand sequences of detected frames only, 21x21 angle matrices from *normalised* (not aspect-corrected)
coordinates, fastdtw with its default distance, and a reference is only comparable when the same hands are present.
"""
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from . import features as F


def legacy_embedding(video):
    """Return (left_seq, right_seq): arrays (frames_with_hand, 441) or None when that hand was never detected."""
    out = []
    n = len(F.HAND_CONNECTIONS)
    for hand in (video.left, video.right):
        present = ~np.isnan(hand[:, 0, 0])
        if not present.any():
            out.append(None)
            continue
        ang = F.hand_angles(np.nan_to_num(hand[present]))          # (frames, 210), normalised coordinates
        full = np.zeros((len(ang), n, n), dtype=np.float32)
        iu = np.triu_indices(n, 1)
        full[:, iu[0], iu[1]] = ang
        full = full + full.transpose(0, 2, 1)
        out.append(full.reshape(len(ang), -1))
    return tuple(out)


def legacy_distance(q, r):
    """Sum of fastdtw distances over the hands present in both; infinite if the hand sets differ."""
    if (q[0] is None) != (r[0] is None) or (q[1] is None) != (r[1] is None):
        return np.inf
    total = 0.0
    for a, b in zip(q, r):
        if a is not None:
            total += fastdtw(a, b)[0]
    return total


def fastdtw_distance(a, b, radius=1):
    return fastdtw(a, b, radius=radius, dist=euclidean)[0]
