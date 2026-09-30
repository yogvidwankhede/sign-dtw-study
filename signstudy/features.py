"""Turn a Video into a (frames, dims) feature sequence. See PREREG.md section 3 for the definitions."""
import numpy as np

FEATURE_SETS = ("F1", "F2", "F3", "F4", "F5", "F6")
POSE_UPPER = (11, 12, 13, 14, 15, 16)  # shoulders, elbows, wrists
# MediaPipe HAND_CONNECTIONS, sorted so the order is deterministic (21 connections)
HAND_CONNECTIONS = tuple(sorted({(0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                                 (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
                                 (19, 20)}))
_IU = np.triu_indices(len(HAND_CONNECTIONS), k=1)


def _scale(arr, size):
    """Normalised image coordinates -> pixel units so angles are not distorted by the frame's aspect ratio."""
    w, h = size
    out = arr.copy()
    out[..., 0] *= w
    out[..., 1] *= h
    out[..., 2] *= w
    return out


def hand_angles(lm):
    """(frames, 21, 3) -> (frames, 210): angle between every pair of connection vectors (upper triangle)."""
    a = np.array([c[0] for c in HAND_CONNECTIONS])
    b = np.array([c[1] for c in HAND_CONNECTIONS])
    vec = lm[:, b, :] - lm[:, a, :]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    unit = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0)
    cos = np.clip(np.einsum("tid,tjd->tij", unit, unit), -1.0, 1.0)
    ang = np.arccos(cos)[:, _IU[0], _IU[1]]
    return np.nan_to_num(ang, nan=0.0)


def hand_coords(lm):
    """Wrist-centred, divided by the wrist-to-middle-finger-base distance. (frames, 63)."""
    rel = lm - lm[:, :1, :]
    scale = np.linalg.norm(rel[:, 9, :], axis=-1)
    scale = np.where(scale > 0, scale, 1.0)
    return (rel / scale[:, None, None]).reshape(len(lm), -1)


def _shoulder_frame(pose):
    """Per-frame shoulder midpoint and width (pixel units); gaps are filled with the video median."""
    mid = (pose[:, 11, :3] + pose[:, 12, :3]) / 2
    width = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1)
    if np.isnan(width).all():
        return None, None
    mid_med, w_med = np.nanmedian(mid, axis=0), np.nanmedian(width)
    mid = np.where(np.isnan(mid), mid_med, mid)
    width = np.where(np.isnan(width) | (width <= 0), w_med, width)
    return mid, width


def _present(hand):
    return ~np.isnan(hand[:, 0, 0])


def _canonical(left, right):
    """Put the hand that moves more in slot 0."""
    def motion(h):
        p = _present(h)
        w = h[:, 0, :2]
        both = p[1:] & p[:-1]
        return float(np.abs(np.diff(w, axis=0))[both].sum()) if both.any() else 0.0
    return (right, left) if motion(right) > motion(left) else (left, right)


def video_features(v, fset="F1", missing="zero", dominant=False):
    """Return (frames, dims) float32 or None when no hand was ever detected."""
    left, right = _scale(v.left, v.size), _scale(v.right, v.size)
    pose = _scale(v.pose, v.size) if fset in ("F4", "F5", "F6") else None
    face = _scale(v.face, v.size) if fset == "F6" else None
    pl, pr = _present(left), _present(right)
    any_hand = pl | pr
    if not any_hand.any():
        return None
    first, last = np.flatnonzero(any_hand)[[0, -1]]
    sl = slice(first, last + 1)
    left, right = left[sl], right[sl]
    if dominant:
        left, right = _canonical(left, right)
    mid = width = None
    if pose is not None:
        mid, width = _shoulder_frame(pose[sl])

    def block(h):
        present = _present(h)
        h0 = np.nan_to_num(h, nan=0.0)
        parts = []
        if fset in ("F1", "F3", "F4", "F5", "F6"):
            parts.append(hand_angles(h0))
        if fset in ("F2", "F3", "F4", "F5", "F6"):
            parts.append(hand_coords(h0))
        if fset in ("F4", "F5", "F6"):
            if mid is None:
                parts.append(np.zeros((len(h), 3)))
            else:
                parts.append((h0[:, 0, :] - mid) / width[:, None])
        f = np.concatenate(parts, axis=1)
        f[~present] = 0.0
        return f, present

    (fl, pl2), (fr, pr2) = block(left), block(right)
    feats = [fl, fr]
    if fset in ("F5", "F6"):
        if mid is None:
            feats.append(np.zeros((len(fl), 3 * len(POSE_UPPER))))
        else:
            pp = _scale(v.pose, v.size)[sl][:, list(POSE_UPPER), :3]
            feats.append(np.nan_to_num((pp - mid[:, None, :]) / width[:, None, None]).reshape(len(fl), -1))
    if fset == "F6":
        if mid is None or np.isnan(face[sl]).all():
            feats.append(np.zeros((len(fl), face.shape[1] * 3)))
        else:
            ff = face[sl]
            feats.append(np.nan_to_num((ff - mid[:, None, :]) / width[:, None, None]).reshape(len(fl), -1))
    x = np.concatenate(feats, axis=1).astype(np.float32)

    if missing == "ffill":
        # carry the last detected hand block forward (leading gaps stay zero)
        d = fl.shape[1]
        for blk, pres in ((slice(0, d), pl2), (slice(d, 2 * d), pr2)):
            last_i = -1
            for t in range(len(x)):
                if pres[t]:
                    last_i = t
                elif last_i >= 0:
                    x[t, blk] = x[last_i, blk]
    elif missing == "drop":
        x = x[pl2 | pr2]
    return x


def resample(x, n=32):
    """Linear interpolation of every dimension to n frames."""
    t = len(x)
    if t == 1:
        return np.repeat(x, n, axis=0)
    src = np.linspace(0, t - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


def dtw_sequence(x):
    """Variable-length sequence used by the DTW methods: every second frame."""
    return x[::2] if len(x) > 1 else x
