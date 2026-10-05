# v0.1.0 (draft; tag only after you have reviewed it)

First release of the study code, pre-registration and aggregate results.

- PopSign ASL v1.0 analysis (pre-registered, tag `prereg-v1`) and ASL Citizen replication (tag `prereg-v2`).
- Learned models (1D-CNN, BiGRU, Transformer) beat DTW 1-NN at every number of examples per sign on both datasets; details in `paper/paper.pdf` and `CLAIMS.md`.
- Exploratory extensions: k-NN voting, Fisher-weighted DTW, MiniRocket, per-k tuning, longer training, soft-DTW prototypes, learned-embedding transfer, hand-slot and movement analyses.
- Reproduce the PopSign tables, figures and numbers from the committed results with `python -m signstudy.report`; CI checks that they match.

Not included: videos, face landmarks, ASL Citizen derived data or per-query rows (licence). Not run: PopSign vocabulary 250, low-power timing.
Isolated-sign recognition only; not translation; no claim of benefit to Deaf users.
