# Pre-registration addendum (written 2026-10-04)

`PREREG.md` (tag `prereg-v1`) stays in force for the PopSign analysis and its results are unchanged. This addendum adds two things,
and says honestly which parts are confirmatory and which are not.

- **Part A is pre-registered.** It is a replication on a second dataset (ASL Citizen). No ASL Citizen video had been scored
  when this was written; only the split files and the licence text had been read, and 3 signs were extracted for a smoke test.
  The commit that adds this file is tagged `prereg-v2` before any ASL Citizen test-split result is produced (test commands
  refuse to run without that tag).
- **Part B is post-hoc and exploratory.** It extends the PopSign study after the main results were known, in response to the
  audit in `REVIEW.md`. Nothing in Part B is a confirmatory test and it carries no multiplicity control.

## Part A. Replication on ASL Citizen (confirmatory)

**Data.** ASL Citizen (Desai et al., NeurIPS 2023 Datasets and Benchmarks; Microsoft Research License: non-commercial research
only, no distribution of the data or modifications, use consistent with participants' consent, destroy personal data when the
research ends). Official splits by participant: train 35, validation 6, test 11 participants, disjoint (checked). Signs: all 2,731
have at least 9 training, 2 validation and 5 test videos and are eligible. Only the videos needed are fetched by HTTP range request
and deleted after landmark extraction. **No face landmarks are extracted or kept.** Landmarks stay in the git-ignored `data/`
directory and are deleted when the research is complete; only aggregate results are committed (no per-query rows, no sign names).
We do not try to identify participants, and make no translation or interpreter claim, as the dataset's recommended-use page asks.

**Vocabularies.** Nested prefixes of a fixed seeded permutation (seed 20260930) of the eligible signs: 10, 50, 100 and 300 signs.

**Examples per sign.** k in {1, 2, 5, 10, all}; "all" is every training video of the sign (about 15). k=20 is dropped because most signs
have fewer than 20 training videos. Draw seeds 0, 1, 2 as in `PREREG.md`.

**Queries.** Every test video of the chosen signs, capped at 3 per (participant, sign); validation queries are built the same
way. Queries are drawn once for the largest vocabulary (300) and restricted for smaller ones.

**Methods and tuning.** Identical to `PREREG.md` sections 3 and 4 (features F1; DTW 1-NN, Euclidean 1-NN, DBA prototype, 1D-CNN,
BiGRU, Transformer; per-family validation selection of missing-hand handling, scaling, DTW band, learning rate, epoch budget),
with these changes: tuning uses the official validation participants at vocabulary 50 and k=10 (seed 0); there is no face or
pose ablation and no augmentation arm; the step rule is unchanged (floor 200, cap 4000).

**Confirmatory hypotheses (replication of the PopSign finding).**
- **H3.** At vocabulary 50 and k in {1, 2, 5}, each learned model has higher top-1 accuracy than DTW 1-NN. Supported only if all
  nine paired signer-cluster 97.5% intervals exclude zero in favour of the learned model.
- **H4.** At vocabulary 50 with all training videos, at least one learned model is above DTW 1-NN (97.5% interval excludes zero,
  and also at the 99.17% level adjusted for three models).
- If either fails we report that the PopSign finding did not replicate, and describe where.

**Exploratory (Part A, not confirmatory).** Vocabularies 100 and 300; the extra baselines of Part B applied to ASL Citizen.

## Part B. PopSign extension (post-hoc, exploratory)

Decided after seeing the PopSign test results and the reviewer-style audit; labelled exploratory everywhere in the paper.

1. **Vocabularies 100 and 250.** These were in `PREREG.md`; they were unfinished only because extraction is slow. Settings tuned
   at 50 signs stay frozen. The condition for a vocabulary is built when it is evaluated, so its test queries are an independent draw
   (the 10- and 50-sign results are not recomputed). Signs beyond the first 50 are extracted without face landmarks (face is
   used only in the 50-sign ablation). Any vocabulary not finished is reported as not run.
2. **Further baselines.** (a) k-NN with kappa > 1: a sign's score is the mean of its kappa smallest reference distances,
   kappa in {1, 3, 5} chosen on validation, for DTW and Euclidean. (b) Per-dimension weighted DTW: features are standardised and
   then multiplied by the square root of a Fisher score (between-sign over within-sign variance of per-video means, from the
   selected references, normalised to mean 1 and clipped to [0.1, 10]); with one reference per sign weights are 1. (c) MiniRocket
   (aeon) with a ridge classifier on 32-frame standardised sequences, 2,000 kernels (10,000 was too slow for the larger conditions),
   no tuning, only for vocabularies up to 100. All on the main setting (F1).
3. **Longer training.** The three learned models with an epoch budget of 200 and a step cap of 12,000, at vocabulary 50.
4. **Per-k tuning.** Every family's validation selection repeated separately for k in {1, 5, 10, all} at vocabulary 50 (k=2 uses the
   k=1 choice and k=20 the k=10 choice), then evaluated on the test split.
5. **"Why" analyses (descriptive).** (a) per-signer share of frames where only MediaPipe's left or only its right hand is detected;
   (b) a slot-swap test: accuracy of DTW and a CNN trained on normal data when the left and right hand slots of the query are
   swapped; (c) per-sign motion (path length of the wrist location in F4, shoulder-width units) against the per-sign gap between
   the CNN and DTW (Spearman correlation with a bootstrap interval over signs).
6. **Intervals in the paper tables** (presentation only).

## Not changing

The main PopSign results, tuned settings, hypotheses H1 and H2, and their outcome (H1 not supported, H2 supported) stay as
reported. If Part B changes the picture, both versions are reported.

## Frozen ASL Citizen settings (chosen on the validation participants at vocabulary 50, k=10, before any ASL Citizen test scoring)

- dtw: {"missing": "ffill", "scale": "raw", "radius": 3}
- euclid: {"missing": "ffill", "scale": "std"}
- cnn1d: {"missing": "ffill", "lr": 0.0003, "epochs": 80}
- bigru: {"missing": "ffill", "lr": 0.001, "epochs": 80}
- transformer: {"missing": "drop", "lr": 0.001, "epochs": 80}

A validation-only dry run of the whole main grid at 50 signs was done to check the code; it showed the same ordering as PopSign on
validation data and its outputs were deleted. The tag `prereg-v2` marks the commit that contains these settings; the ASL
Citizen test commands refuse to run without it.

## Part B, second batch (added 2026-10-05; post-hoc, exploratory; written before these analyses were run)

These two baselines answer the audit's remaining requests. Test results of the other methods are already known on both datasets, so nothing here is
confirmatory.

7. **Learned embedding with nearest-neighbour search (transfer from other signs).** A 1D-CNN (the main architecture, tuned settings of the CNN
   family) is trained as a classifier on the training videos of *other* signs (ASL Citizen: signs 101-300 of the fixed order; PopSign: signs 51-100 if
   extracted), using all their training examples. Its pooled hidden features (before the linear head) are then used as an embedding: a test query of
   a sign among the first 50 is assigned to the sign of its nearest reference in embedding space (Euclidean after L2 normalisation), with k references per
   sign drawn exactly as for the other methods. No training example of the evaluated signs is used to train the embedding. Reported for k in the usual set,
   vocabulary 50, 3 seeds (embedding initialisation and reference draws).
8. **Soft-DTW prototypes.** One soft-DTW barycentre per sign (tslearn 0.9.0 `softdtw_barycenter`) from the sign's references resampled to 32 frames, gamma chosen on
   validation from {0.1, 1.0} at vocabulary 50 and k=10, then nearest prototype by exact DTW as for the DBA prototypes. Only vocabularies 10 and 50 and k up to 10
   (barycentre fitting is slow); other conditions are reported as not run.
9. **PopSign vocabulary 100.** Extraction was extended to 100 signs. The main grid for vocabulary 100 uses the frozen settings of `prereg-v1` (as
   `PREREG.md` planned) and is evaluated on queries drawn when the 100-sign condition is built (independent of the 10- and 50-sign draws). Vocabulary 250 stays not run.

### Outcome of the second batch (2026-10-06)
Items 7 and 8 were run on ASL Citizen (embedding transfer, soft-DTW) and soft-DTW on PopSign. **Item 9 (PopSign vocabulary 100) was not completed:** extraction reached 85 of
100 signs and then crashed with a write error because the disk was full (other processes on the machine had filled it; only about 4 GB were free afterwards, too little to
continue). PopSign vocabulary 100, and the PopSign version of the embedding baseline (which needs signs 51-100), are therefore reported as not run.
