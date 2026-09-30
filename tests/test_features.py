import numpy as np
import pytest

from signstudy import features as F
from signstudy.dataset import Video


def _hand(rng, t=12):
    base = rng.normal(size=(21, 3))
    return base[None] + 0.05 * rng.normal(size=(t, 21, 3))


def _video(left, right=None, t=12, size=(100, 100)):
    rng = np.random.default_rng(1)
    right = np.full((t, 21, 3), np.nan, dtype=np.float32) if right is None else right
    pose = np.full((t, 33, 4), np.nan, dtype=np.float32)
    return Video(sign="x", split="train", signer="s", name="n", left=left.astype(np.float32),
                 right=right.astype(np.float32), pose=pose, face=None, size=size)


def test_angles_have_210_dims_and_range():
    rng = np.random.default_rng(0)
    a = F.hand_angles(_hand(rng))
    assert a.shape == (12, 210)
    assert (a >= 0).all() and (a <= np.pi + 1e-6).all()


def test_angles_invariant_to_translation_scale_and_rotation():
    rng = np.random.default_rng(0)
    h = _hand(rng)
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    g = (h @ rot.T) * 2.5 + np.array([3.0, -1.0, 2.0])
    assert np.allclose(F.hand_angles(h), F.hand_angles(g), atol=1e-4)


def test_upper_triangle_distance_is_proportional_to_full_matrix():
    """Euclidean distance on the original 21x21 matrix equals sqrt(2) times the upper-triangle distance."""
    rng = np.random.default_rng(0)
    a, b = F.hand_angles(_hand(rng, 1)), F.hand_angles(_hand(rng, 1))
    n = len(F.HAND_CONNECTIONS)

    def full(v):
        m = np.zeros((n, n))
        m[np.triu_indices(n, 1)] = v
        return (m + m.T).ravel()

    assert np.isclose(np.linalg.norm(full(a[0]) - full(b[0])), np.sqrt(2) * np.linalg.norm(a[0] - b[0]))


def test_no_hand_returns_none():
    nan = np.full((8, 21, 3), np.nan)
    assert F.video_features(_video(nan, t=8)) is None


def test_trim_and_zero_fill():
    rng = np.random.default_rng(0)
    left = _hand(rng, 10)
    left[:3] = np.nan
    left[8:] = np.nan
    x = F.video_features(_video(left, t=10))
    assert x.shape == (5, 420)
    assert np.abs(x[:, 210:]).sum() == 0  # right hand never seen


def test_ffill_and_drop_change_length_or_values():
    rng = np.random.default_rng(0)
    left = _hand(rng, 10)
    left[4:6] = np.nan
    z = F.video_features(_video(left, t=10), missing="zero")
    f = F.video_features(_video(left, t=10), missing="ffill")
    d = F.video_features(_video(left, t=10), missing="drop")
    assert np.abs(z[4:6]).sum() == 0 and np.allclose(f[4], f[3])
    assert len(d) == len(z) - 2


def test_aspect_correction_uses_frame_size():
    rng = np.random.default_rng(0)
    h = np.abs(_hand(rng)) % 1
    wide = F.video_features(_video(h, size=(200, 100)))
    tall = F.video_features(_video(h, size=(100, 200)))
    assert not np.allclose(wide, tall)


def test_resample_length_and_endpoints():
    x = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
    r = F.resample(x, 32)
    assert r.shape == (32, 3) and np.allclose(r[0], x[0]) and np.allclose(r[-1], x[-1])
    assert F.resample(x[:1], 32).shape == (32, 3)


@pytest.mark.parametrize("fset,dims", [("F1", 420), ("F2", 126), ("F3", 546), ("F4", 552), ("F5", 570)])
def test_feature_dims(fset, dims):
    rng = np.random.default_rng(0)
    v = _video(_hand(rng), _hand(rng))
    v.pose = np.nan_to_num(np.random.default_rng(2).random((12, 33, 4))).astype(np.float32)
    assert F.video_features(v, fset).shape[1] == dims
