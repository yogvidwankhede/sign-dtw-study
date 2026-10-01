# Pre-registration: when does training-free DTW match learned models on hand landmarks?

Written before any test-split accuracy was computed. Git history is the timestamp: the commit that adds this file
(and the tag `prereg-v1` on the commit that freezes the code) must precede every file in `results/` that is labelled
"test". Anything changed afterwards is listed under **Deviations** at the bottom, with a date and a reason.

## 0. What this study is and is not

- It measures **isolated-sign recognition** from MediaPipe landmarks on one dataset of one sign language (**American
  Sign Language, smartphone selfie videos, PopSign ASL v1.0**). It is not sign language translation or interpretation.
  Hands dominate the features; non-manual markers are not modelled.
- It is an empirical comparison, not a new method. No claim of state of the art and no claim about benefit to Deaf users.
- Expected from prior time-series work (Bagnall & Lines 2014): tuned nearest-neighbour DTW is strong with few examples.
  We quantify where that holds for signs, and where learned models overtake it.
- The study is not Deaf-led. See the ethics section of the paper for what that limits.

## 1. Data

- PopSign ASL v1.0, category `game` only (the example variant of each of 250 signs), CC BY 4.0, Georgia Tech / DPAN.
  Videos are downloaded for extraction and deleted; only derived landmarks are kept.
- Splits are the dataset's own signer-disjoint splits (train 31, val 8, test 8 signers). The code asserts that no signer
  appears in two splits and aborts otherwise. The signer id is the filename prefix before the first `-`.
- Vocabulary: nested subsets of the first N signs of a fixed order (seeded permutation of the sorted sign list,
  seed 20260930): N in {10, 50, 100, 250}. (PopSign has 250 signs, so the 300-sign condition is not possible here.)
  A vocabulary condition is reported only if all its signs were extracted; unfinished conditions are reported as not run.
- References / training examples per sign: k in {1, 2, 5, 10, 20, full}. Examples are drawn at random without
  replacement from the train split, nested (k=1 subset of k=2 ...), with draw seed s in {0, 1, 2}. "full" is every
  train video of the sign.
- Test queries: up to 3 videos per (test signer, sign), drawn once with a fixed seed and shared by every method.
  Validation queries are built the same way from the val split. Validation signers are the only data used to choose
  hyperparameters; test signers are not used for any choice.
- Videos in which no hand is ever detected are kept and counted as errors (the count is reported).

## 2. Landmark extraction (fixed)

MediaPipe Holistic (legacy solution), `mediapipe==0.10.14`, model_complexity 1, smooth_landmarks on,
detection/tracking confidence 0.5, frames resized so the short side is 480 px (INTER_AREA). x and y are multiplied by
the frame width and height before any angle or distance is computed (the original repository used normalised
coordinates, which distorts angles on non-square frames). Face landmarks are a local cache only and are not released.
Hand and pose landmarks may be released with attribution under CC BY 4.0.

## 3. Features

Per hand, per frame. A sequence is trimmed from the first to the last frame where any hand was detected.

| id | content |
|---|---|
| F1 | angles between all pairs of the 21 hand connections (210 per hand; identical information to the original 21x21 matrix, and Euclidean distances differ only by a constant factor sqrt 2) |
| F2 | hand shape: landmarks relative to the wrist, divided by the wrist-to-middle-finger-base distance (63 per hand) |
| F3 | F1 + F2 |
| F4 | F3 + hand location: wrist relative to the shoulder midpoint in shoulder-width units (3 per hand) |
| F5 | F4 + upper-body pose (shoulders, elbows, wrists; same normalisation) |
| F6 | F5 + face contour landmarks (128 points, same normalisation); ablation only, never released |

Main grid: F1 for every method (the repository's existing features). Feature ablations (F1-F6, every method of section 4 except FastDTW, legacy and soft-DTW) run at vocabulary 50 and k in {5, full},
each family keeping its tuned scaling, band and learning rate.

Missing hand in a frame: zero-fill, carry the last detection forward (ffill), or drop frames with no hand; chosen per family
on validation (section 4) and also reported as an ablation.
Handedness: main setting uses MediaPipe's left/right slots as given; ablation puts the hand with the larger total
wrist displacement in slot 1 (dominant-hand canonicalisation). One- versus two-handed: a sign is "two-handed" if both
hands are detected in at least 50% of the hand-present frames across its train videos; accuracy is reported by group.

Scaling: the learned models standardise each dimension with the mean and standard deviation of their own training
subset (never val or test data). DTW and Euclidean 1-NN get the same standardisation as an option (dividing by the
standard deviation of the selected references; centring cancels in differences) and keep it only if validation prefers it.

Temporal handling: DTW methods use the trimmed sequence downsampled by 2 (variable length). Euclidean nearest
neighbour and the learned models use the trimmed sequence linearly resampled to 32 frames.

## 4. Methods

Each family picks its own preprocessing on **validation signers only** (vocabulary 50, k=10, draw seed 0), from the
grids below, so that no method is handicapped by an arbitrary default. Ties go to the earlier option in the order listed.
The features themselves stay F1 in the main grid.

| family | chosen on validation from |
|---|---|
| DTW | missing hands {zero, ffill, drop} x scaling {raw, per-dimension standard deviation of the selected references} x Sakoe-Chiba radius {3, 5, 8, unconstrained} frames |
| Euclidean-1NN | missing hands x scaling |
| 1D-CNN, BiGRU, Transformer | missing hands x learning rate {1e-3, 3e-4} x epoch budget {20, 80} |

1. **DTW-1NN**: DTW with Euclidean local cost, nearest reference decides (own compiled implementation, tested against
   tslearn 0.9.0). The naive default (zero-fill, raw, unconstrained) is also reported as a reference row.
2. **FastDTW-1NN** (radius 1, fastdtw 0.3.4): latency and accuracy only, on a fixed subsample (vocabulary 50, k=10).
3. **Legacy DTW**: the original repository's pipeline (per-hand DTW over detected frames only, hand-set must match,
   normalised coordinates, FastDTW), as a reference point.
4. **DBA prototype**: one DTW barycentre per sign from its k references (tslearn 0.9.0 default initialisation, barycentre length = median length of the
   sign's references, 30 iterations), nearest prototype by DTW, using the DTW family's chosen preprocessing.
5. **Soft-DTW prototype**: exploratory, run only if time allows; otherwise reported as not run.
6. **Euclidean-1NN** on the 32-frame resampled flattened features.
7. **1D-CNN**, 8. **BiGRU**, 9. **small Transformer encoder** (architectures in `signstudy/models.py`): cross-entropy
   with label smoothing 0.05, AdamW, one-cycle schedule, batch 64, and a fixed number of gradient steps
   steps = min(max(ceil(epochs x N / 64), 200), 4000) for N training examples (no early stopping, so validation labels
   are not used for training). The 4000-step cap means the largest conditions get only a few epochs; this is a limitation.
   Main grid has no data augmentation; one extra arm with light augmentation (time stretch, noise, scale) at vocabulary 50
   checks that the learned models are not a strawman. Seeds 0, 1, 2 (subset draw and initialisation).

## 5. Metrics and uncertainty

- Top-1 and top-5 accuracy on the test queries. Per-seed values are averaged over seeds first.
- Confidence intervals: cluster bootstrap over the 8 test signers (10,000 resamples, signers resampled with
  replacement, all their queries kept). Differences between methods use the same resamples (paired).
  With 8 clusters the intervals are wide; we report them as they are and do not use p-values.
- Seed variation (training noise) is reported separately as the range across the 3 seeds.
- Per-sign accuracy, and the most frequent confusions. PopSign has no phonological annotation, so we do not call any
  pair "phonologically similar"; we list data-driven top confusions and leave linguistic interpretation to a signer.

## 6. Hypotheses, fixed in advance

- **H1 (confirmatory)**: at vocabulary 50 and k <= 5, DTW-1NN has higher top-1 accuracy than each learned model.
- **H2 (confirmatory)**: at vocabulary 50 and k = full, at least one learned model has higher top-1 accuracy than DTW-1NN.
- H1 and H2 are each judged with 97.5% paired-bootstrap intervals (two confirmatory tests, Bonferroni). A difference
  counts as supported only if the interval excludes 0; otherwise we report "not distinguishable at this sample size".
- Everything else (crossover k* per vocabulary, feature ablations, DTW variants, latency) is **exploratory**, reported
  with 95% intervals and labelled as such. No multiplicity correction is applied to exploratory comparisons.
- Expectation written down before results: DTW is strongest at k of 1 to 5; learned models overtake it as k and the
  vocabulary grow. We will report it if this is false.

## 7. Timing

Per-query latency (median and p95) and CPU time on the M4 Max, single thread and all cores, for DTW-1NN, FastDTW,
DBA prototype, Euclidean-1NN and each learned model at vocabulary 50. No low-power device is available; that row is
reported as "not measured" unless one is obtained.

## 8. Secondary replication (planned, may not be completed)

ASL Citizen, private landmarks only (its licence forbids redistributing data or modifications), vocabulary 50,
k in {1, 5, 20, full}, same methods. If not completed it is reported as not done.

## 9. Decided before seeing results vs after

Before: everything above. After (to be logged under Deviations): anything not listed here, including any bug fix that
changes a number.

## Deviations

Revisions made after a validation-only smoke test on 4 signs and **before any test-split scoring** (30 Sep 2026):

1. Added per-dimension standardisation as a DTW/Euclidean option, because the learned models are standardised and a
   scale mismatch could have handicapped DTW.
2. Added missing-hand handling (zero / ffill / drop) to the validation selection for every family, because the smoke test
   suggested zero-filling creates jumps DTW cannot ignore while the original repository dropped those frames.
3. Replaced the 600-step floor of the learned-model step rule by 200, and the epoch grid by {20, 80}, because with the
   floor at 600 the epoch choice had no effect for datasets under about 1,300 examples.
4. DBA uses tslearn's default deterministic initialisation (the installed version has no random initialisation option) and a
   barycentre length equal to the median reference length: the tslearn default (longest reference) made prototypes
   unmatched to shorter queries and collapsed accuracy to chance in the validation dry run.
6. Validation-only development runs (4 and 10 signs, validation split) were committed to git history before the final
   tuning; they contain no test-split scores. They were removed from the working tree before tuning at 50 signs.
5. Clarified that the "all methods" ablation covers the methods of section 4 (the original sentence pointed at section 7).

Post-freeze notes (made after test results were seen; every one is listed here):

7. The feature-set ablation (F1-F6, handedness) used zero-filled missing hands, because the ablation stage builds its
   conditions with the default setting; the pre-registration did not state this. The F1/drop row of the same table is the
   main setting. Compare rows within the table.
8. Exploratory supplement, added after seeing that DTW trailed the learned models: F3, F4, F5 and F4 with dominant-hand
   slots under the tuned drop setting (`pipeline ablate_drop`), run at vocabulary 50, k in {5, all}. It is exploratory and
   was added so that DTW's best case is not understated.
9. The timing script's CPU-load column was found to include background BLAS threads, so timing was rerun with all thread
   limits pinned; only the rerun is reported. Latency was measured with other processes running on the machine.
10. History rewrite (1 Oct 2026, before anything was pushed or published): feature-cache pickles (`data/features/*.pkl`, about 2 GB)
    had been committed by mistake and were removed from every commit with `git filter-branch`. The tag `prereg-v1` was
    rewritten with them: it pointed to commit `e9fc22e` and now points to `dbbcc2b`. The tagged tree is identical to the original
    in the content of every file except the removed cache files and one LaTeX build log (checked blob by blob).
