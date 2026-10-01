"""Write a releasable copy of the PopSign-derived landmarks: hands and upper-body pose only, no face, no video.

Usage: python scripts/package_landmarks.py OUT_DIR
Produces OUT_DIR/<sign>/<split>.npz (float16), README.md with attribution and licence, and the extraction log.
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signstudy import config as C  # noqa: E402

README = """# PopSign ASL v1.0: MediaPipe hand and pose landmarks (derived data)

Derived from PopSign ASL v1.0 (Starner et al., NeurIPS 2023 Datasets and Benchmarks), Georgia Institute of Technology
and the Deaf Professional Arts Network, licensed CC BY 4.0. This derived data is released under CC BY 4.0; please
attribute Georgia Tech and DPAN and cite the PopSign paper.

Contents: per sign and split, an .npz with `offsets`, `names` (original file names, which contain anonymised signer ids),
`sizes` (frame width/height used, needed to undo the aspect ratio), `left`, `right` (21x3 hand landmarks), `pose`
(33x4). Values are float16 and NaN where a part was not detected. Landmarks come from MediaPipe Holistic 0.10.14, model
complexity 1, frames resized to a 480 px short side. Face landmarks and videos are not included.

Use is limited by the PopSign dataset card: isolated sign recognition research; not continuous signing, translation, or
identification of participants.
"""


def main(out):
    out = Path(out)
    for f in sorted(C.LANDMARKS.glob("*/*.npz")):
        dest = out / f.parent.name / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        z = np.load(f)
        np.savez_compressed(dest, **{k: z[k] for k in ("offsets", "names", "sizes", "left", "right", "pose")})
    (out / "README.md").write_text(README)
    log = C.LANDMARKS / "EXTRACTION_LOG.jsonl"
    if log.exists():
        shutil.copy(log, out / "EXTRACTION_LOG.jsonl")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
