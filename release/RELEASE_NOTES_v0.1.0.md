# v0.1.0

First release of the study code, pre-registration and aggregate results.

- PopSign ASL v1.0 analysis (pre-registered, tag `prereg-v1`) and ASL Citizen replication (tag `prereg-v2`).
- Learned models (1D-CNN, BiGRU, Transformer) are ahead of DTW 1-NN in nearly every setting on both datasets (with one example per sign at 250 PopSign signs every method is near 10% and the BiGRU is below DTW); details in `paper/paper.pdf` and `CLAIMS.md`.
- Exploratory extensions: k-NN voting, Fisher-weighted DTW, MiniRocket, per-k tuning, longer training, soft-DTW prototypes, learned-embedding transfer, hand-slot and movement analyses.
- Reproduce the PopSign tables, figures and numbers from the committed results with `python -m signstudy.report`; CI checks that they match.

Not included: videos, face landmarks, ASL Citizen derived data or per-query rows (licence). PopSign vocabularies up to the full 250 signs were run (exploratory beyond 50 signs). Not run: low-power timing; several exploratory baselines at 250 signs.
Isolated-sign recognition only; not translation; no claim of benefit to Deaf users.
