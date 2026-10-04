# When does training-free DTW match learned models on hand landmarks?

A pre-registered comparison of nearest-neighbour dynamic time warping (DTW) with a 1D-CNN, a BiGRU and a small Transformer,
for isolated sign recognition from MediaPipe landmarks on PopSign ASL v1.0, with signer-independent splits, varying numbers of
references per sign and vocabulary sizes; plus a pre-registered replication on ASL Citizen (up to 300 signs) and exploratory
extensions (`PREREG_ADDENDUM.md`). It is not sign language translation, and it makes no claim about benefit to Deaf users.

Status: see `PREREG.md` (the plan, written before test results), `CLAIMS.md` (where each number comes from), and
`HUMAN_TODO.md`. Nothing here has been published or pushed.

## Reproduce

```
make setup                 # Python 3.11 venv, pinned requirements
make test                  # unit tests (DTW engine vs tslearn, features, statistics)
make quick VOCAB=10        # download -> landmarks -> tune on validation -> test grid -> ablations -> timing -> tables/figures
make report                # regenerate every table, figure and paper/numbers.tex from results/raw
SIGNSTUDY_DATASET=aslc python -m signstudy.aslc prepare   # ASL Citizen: fetch split files; then `... aslc extract 300`, `... pipeline tune|main`
make paper                 # compile paper/paper.tex (tectonic)
```

`make quick VOCAB=50` and larger need more download time: each sign is about 4.5 GB of video, streamed, converted to
landmarks and deleted. The landmark cache lives in `data/landmarks/` and reruns skip finished signs. Test-split commands refuse
to run unless the git tag `prereg-v1` exists, is an ancestor of HEAD, and the code is committed.

Reproducibility checks: `python -m signstudy.report` from a clean clone regenerates every table, figure, number and the
claims ledger byte-identically from the committed `results/` (this is what CI checks); `python scripts/check_rerun.py` recomputes
test rows from the landmark cache and compares them with the stored ones (DTW, Euclidean and one CNN run matched exactly on
the machine that produced them; other hardware may differ for the learned models). Test queries are drawn once for the largest
vocabulary and restricted to smaller ones, so a smaller vocabulary run on its own would draw different queries.

## Layout

- `signstudy/` code: `extract.py` (download + MediaPipe), `features.py`, `dtwlib.py` (numba DTW checked against tslearn),
  `models.py`/`train.py`, `run.py`/`pipeline.py` (grids), `stats.py` (signer-cluster bootstrap), `timing.py`, `report.py`.
- `results/raw/` per-query results with provenance JSON; `results/tables/` generated tables; `figures/` generated figures.
- `DATA.md` licences and handling; `ETHICS.md` scope and community feedback; `VENUE.md`; `LLM_DISCLOSURE.md`.

## Licence

Code: MIT (proposed). Derived hand/pose landmarks: CC BY 4.0 with attribution to Georgia Tech and DPAN (PopSign).
