"""Fixed constants for the study. Everything a result depends on is set here or in PREREG.md."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LANDMARKS = DATA / "landmarks"
DISTANCES = DATA / "distances"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATASET = "popsign_v1_0"
CATEGORY = "game"
SPLITS = ("train", "val", "test")
BASE_URL = "https://signdata.cc.gatech.edu/data"

MASTER_SEED = 20260930
MEDIAPIPE_VERSION = "0.10.14"
SHORT_SIDE = 480  # frames are resized so the short side is this many pixels before MediaPipe

# Face landmarks are kept only as a local cache (never released); this is the set of contour vertices.
VOCAB_SIZES = (10, 50, 100, 250)
REFS_PER_SIGN = (1, 2, 5, 10, 20, "full")
SEEDS = (0, 1, 2)
TEST_PER_SIGNER_SIGN = 3   # test videos drawn per (test signer, sign)


def sign_list():
    """All 250 signs in the fixed pre-registered order (seeded permutation of the sorted list)."""
    signs = sorted((ROOT / "signs_game.txt").read_text().split())
    assert len(signs) == 250
    order = np.random.default_rng(MASTER_SEED).permutation(len(signs))
    return [signs[i] for i in order]


def vocab(n):
    """Nested vocabularies: the first n signs of the fixed order."""
    return sign_list()[:n]
