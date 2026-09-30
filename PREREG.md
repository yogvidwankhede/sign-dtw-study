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

Main grid: F1 for every method (the repository's existing features). Feature ablations (F1-F6, all methods listed in
section 7) run at vocabulary 50 and k in {5, full}.

Missing hand in a frame: main setting is zero-fill. Ablation: carry the last detection forward; drop the frame.
Handedness: main setting uses MediaPipe's left/right slots as given; ablation puts the hand with the larger total
wrist displacement in slot 1 (dominant-hand canonicalisation). One- versus two-handed: a sign is "two-handed" if both
hands are detected in at least 50% of the hand-present frames across its train videos; accuracy is reported by group.

Scaling: DTW and Euclidean nearest neighbour use the raw feature values. The learned models standardise each
dimension with the mean and standard deviation of their own training subset (never val or test data).

Temporal handling: DTW methods use the trimmed sequence downsampled by 2 (variable length). Euclidean nearest
neighbour and the learned models use the trimmed sequence linearly resampled to 32 frames.

## 4. Methods

1. **DTW-1NN**: DTW with Euclidean local cost, nearest reference decides (tslearn 0.9.0). Sakoe-Chiba radius chosen on
   validation from {3, 5, 8, unconstrained} frames at vocabulary 50, k=10, draw seed 0 (ties go to the smaller radius).
2. **FastDTW-1NN** (radius 1, fastdtw 0.3.4): latency and accuracy only, on a fixed subsample (vocabulary 50, k=10).
3. **Legacy DTW**: the original repository's pipeline (per-hand DTW over detected frames only, hand-set must match,
   normalised coordinates, FastDTW), as a reference point.
4. **DBA prototype**: one DTW barycentre per sign from its k references (30 iterations), nearest prototype decides.
5. **Soft-DTW prototype**: soft-DTW barycentre per sign (gamma chosen on validation from {0.1, 1.0}); exploratory.
6. **Euclidean-1NN** on the 32-frame resampled flattened features.
7. **1D-CNN**, 8. **BiGRU**, 9. **small Transformer encoder** (architectures in `signstudy/models.py`): cross-entropy,
   AdamW, a fixed number of gradient steps (no early stopping, so validation labels are not used for training),
   hyperparameters chosen on validation at vocabulary 50, k=10, seed 0 from a small fixed grid, then frozen.
   Main grid has no data augmentation; one extra arm with light augmentation (time stretch, noise, hand scale) at
   vocabulary 50 checks that the learned models are not a strawman.
   Seeds 0, 1, 2 (subset draw and initialisation).

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

(none yet)
