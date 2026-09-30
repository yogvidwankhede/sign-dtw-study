"""Fast DTW distance matrices (numba, parallel over query/reference pairs).

Local cost is squared Euclidean distance; the distance returned is sqrt of the accumulated cost, which matches
tslearn.metrics.dtw (checked in tests/test_dtw_engine.py). The optional Sakoe-Chiba band is centred on the scaled
diagonal j = i*(m-1)/(n-1), so sequences of different length are compared fairly; with equal lengths it is the usual band.
"""
import numpy as np
from numba import njit, prange


def pad(seqs):
    """List of (len, D) arrays -> (N, Lmax, D) float32 array and lengths."""
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    out = np.zeros((len(seqs), int(lens.max()), seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lens


@njit(fastmath=True, cache=True)
def _dtw_pair(a, n, b, m, radius):
    inf = np.float32(np.inf)
    prev = np.full(m + 1, inf, dtype=np.float32)
    cur = np.full(m + 1, inf, dtype=np.float32)
    prev[0] = 0.0
    d = a.shape[1]
    for i in range(1, n + 1):
        cur[:] = inf
        if radius >= 0:
            centre = (i - 1) * (m - 1) / (n - 1) if n > 1 else 0.0
            lo = max(1, int(np.ceil(centre - radius)) + 1)
            hi = min(m, int(np.floor(centre + radius)) + 1)
        else:
            lo, hi = 1, m
        for j in range(lo, hi + 1):
            s = np.float32(0.0)
            for k in range(d):
                diff = a[i - 1, k] - b[j - 1, k]
                s += diff * diff
            best = prev[j]
            if prev[j - 1] < best:
                best = prev[j - 1]
            if cur[j - 1] < best:
                best = cur[j - 1]
            cur[j] = s + best
        prev, cur = cur, prev
    return np.sqrt(prev[m])


@njit(parallel=True, fastmath=True, cache=True)
def _dtw_matrix(Q, ql, R, rl, radius):
    nq, nr = Q.shape[0], R.shape[0]
    out = np.empty((nq, nr), dtype=np.float32)
    for idx in prange(nq * nr):
        q = idx // nr
        r = idx % nr
        out[q, r] = _dtw_pair(Q[q], ql[q], R[r], rl[r], radius)
    return out


def dtw_matrix(queries, refs, radius=-1):
    """Distance matrix (len(queries), len(refs)); radius in frames, negative = unconstrained."""
    Q, ql = pad(queries)
    R, rl = pad(refs)
    return _dtw_matrix(Q, ql, R, rl, np.int64(radius))


def dtw_pair(a, b, radius=-1):
    return float(dtw_matrix([a], [b], radius)[0, 0])


def euclid_matrix(Xq, Xr):
    """Plain Euclidean distance between flattened fixed-length sequences, (Q, R)."""
    Xq, Xr = Xq.reshape(len(Xq), -1).astype(np.float32), Xr.reshape(len(Xr), -1).astype(np.float32)
    d2 = (Xq ** 2).sum(1)[:, None] + (Xr ** 2).sum(1)[None, :] - 2 * Xq @ Xr.T
    return np.sqrt(np.maximum(d2, 0))
