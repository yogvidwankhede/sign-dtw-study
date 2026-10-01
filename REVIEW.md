# Adversarial review and reproducibility audit

Written by the assistant that helped build the study, as an adversarial reviewer and auditor. Run on 2026-10-01 against
`study-setup` HEAD. Venue forms were fetched on 2026-09-30 from the ACL Rolling Review site (review form and Responsible NLP
checklist); no suitable venue has an open call yet (`VENUE.md`), so ARR's form is used as the closest proxy for ACL-family
sign-language workshops. The author must re-check the real form when a call is posted.

**Recommendation: FIX-FIRST** (not "do not submit"). Priority list at the end.

---

## Step 1. Reviewer report (ARR review-form fields)

**Paper summary.** A pre-registered empirical comparison of nearest-neighbour DTW with Euclidean 1-NN, DTW prototypes and three
small learned models (1D-CNN, BiGRU, Transformer) for isolated sign recognition from MediaPipe landmarks on PopSign ASL v1.0, with
signer-disjoint splits, 1 to all examples per sign, and vocabularies of 10 and 50 signs. The learned models beat DTW at every number
of examples per sign (36 of 36 paired comparisons favour them, all with 95% signer-cluster intervals excluding zero), refuting the
authors' own pre-registered expectation that DTW wins with few examples. Preprocessing choices are tuned per method family on
validation signers; DTW's best case across feature ablations reaches 71% against 93% for the best CNN.

**Summary of strengths.**
1. Honest negative result against the authors' own stated hypothesis, with the hypothesis fixed in advance and the revisions listed.
2. Signer-independent evaluation with signer-clustered intervals and paired differences; leakage checked (signers disjoint, no
   duplicate sequences or file names across splits).
3. Fair treatment of the baselines: each family chose its preprocessing on validation signers, and the paper shows how much a naive
   default would have handicapped DTW (11.4 points on validation).
4. Very good reproducibility: pinned environment, cached landmarks, per-result provenance, generated numbers and claims ledger,
   clean-clone regeneration of every table and figure, DTW engine checked against tslearn.
5. Careful scope and ethics language; dataset used with a licence that permits derived data.

**Summary of weaknesses.**
1. **Narrow evidence.** One dataset, one language, one-handed smartphone video, at most 50 signs. The study's own plan promised
   vocabularies up to 250-300 and an ASL Citizen replication; neither was run. "Vocabulary size" is therefore barely varied
   (10 vs 50). The title's generality ("when does DTW...") exceeds what two vocabularies on one dataset can support.
2. **The main comparison uses features that may disadvantage template matching.** F1 hand-angle features are invariant to
   hand location, orientation and scale, so they discard information that distinguishes many signs. The ablation (F4, F5) helps both
   families, but hyperparameters and preprocessing were tuned only at F1; DTW was not re-tuned at the richer feature sets.
3. **Tuning protocol.** All choices were made at one condition (50 signs, 10 examples per sign, one draw) and applied to every
   k and vocabulary. The optimum at k=1 or "all" can differ (for DTW the band and scaling; for models the epoch budget). Every
   learned model picked the longest epoch budget in its grid (grid edge), and a step cap of 4000 gives the largest conditions only a
   few epochs, so "all examples" results may understate the models, while the DTW family had a larger option grid (24 vs 12).
4. **Eight test signers.** Intervals are wide (15-20 points); the 36 learned-vs-DTW comparisons are highly dependent, so "36 of 36"
   overstates independent evidence. The seed range (at most 3.4 points) is small next to the signer intervals, so conclusions about
   close pairs (Euclidean vs DTW; the three models among themselves) are not resolvable.
5. **Post-hoc elements.** Ablation maxima are taken on the test signers (optimistic for both families); the drop-frames
   feature supplement was added after seeing results (disclosed); the pre-registration is a git history, not an independent
   registry, and the history was rewritten once to remove accidentally committed cache files (disclosed, tagged tree verified
   identical apart from removed files).
6. **No explanation.** The paper says why DTW trails is untested. For a methods paper that is a gap: no analysis of alignment
   behaviour, per-dimension weighting, or whether errors concentrate on signs distinguished by location/movement.
7. **Timing is thin.** Shared machine, no low-power device, compiled exact DTW vs pure-Python FastDTW is not an algorithmic
   comparison, and CPU-load figures are not reported.
8. **Not Deaf-led, community feedback pending.** The paper is candid about it, but the feedback step is still undone.

**Comments, suggestions, typos.** Report top-5 in the main text (it is in the results files). Give sign-level results by the
location/movement type if annotation is available. State the PopSign v2.0 existence (FG 2026) and why v1.0 was used. The abstract
could state the sample of 50 signs and 8 test signers.

**Scores (ARR scales).** Soundness 3.5/5 (claims match the evidence; protocol limitations above). Excitement 2/5, "Potentially
Interesting" (useful negative result, modest scope). Overall assessment 3/5, "Findings" (a workshop paper). Reviewer confidence 4/5.
Reproducibility 5/5. Datasets 4/5. Software 4/5. Author identity knowledge 5/5 (the draft is not anonymised).

**Limitations and societal impact.** Adequate and candid. **Ethical concerns:** none that need ethics review; the data are an
existing consented dataset. The missing Deaf-community feedback should be filled in or stated as not done.

**Missing baselines** (verified references): ROCKET-style random-convolution classifiers (Dempster et al. 2020,
doi:10.1007/s10618-020-00701-z), the bake-off of time-series classifiers (Bagnall et al. 2016/17,
doi:10.1007/s10618-016-0483-9) - both now named in the paper's limitations; kNN on learned embeddings (Vandendriessche et al.
2025, arXiv:2502.20171); multivariate DTW variants (dependent vs independent), shapeDTW, derivative DTW; metric learning or
per-dimension weighting for DTW; k-NN with k>1; pre-trained pose models.

**Missing related work.** Prior sign recognition on PopSign itself (the dataset paper's LSTM baseline is cited as data, not
compared); the PopSign v2.0 paper (Crossref: doi:10.1109/fg67764.2026.11556933, seen in a lookup, not read, so not cited).

**What would change my recommendation.** To "Conference"-level: results on a second dataset (even ASL Citizen at 50 signs),
vocabularies to 250, DTW and the models re-tuned at each k, a stronger DTW/time-series baseline set, and an analysis explaining the
gap.

### The harsh "Reviewer 2" version

> The authors set out to show DTW is competitive and found that it isn't, then spent the remainder of the paper arguing
> the baseline was tuned fairly. The evidence is 50 signs, 8 test signers and one dataset. The "vocabulary sweep" has two points. All
> neural models were tuned at a single condition and all picked the edge of their grid, so the headline numbers are neither
> converged nor characteristic. DTW is evaluated on 420 noisy angle features that throw away location, then tuned over a
> hand-picked option set; no obvious DTW improvement (weighting, derivative DTW, shapeDTW, k>1) is tried. The "pre-registration"
> is a git repository that was edited, deleted from, rewritten and annotated after the results were seen. The paper
> explicitly does not know why its main result holds. With eight clusters, "all 36 intervals exclude zero" is a repeated look at
> the same eight signers. I would not accept this beyond a workshop poster. Not a methods contribution; a short negative report.

---

## Step 2. Claims audit

Every empirical number in the paper text is a LaTeX macro generated by `python -m signstudy.report` and listed in `CLAIMS.md`
(645 entries: claim -> results file -> script). The numbers in tables and the figure are generated by the same script. A clean
clone regenerates `results/tables`, `paper/numbers.tex`, `CLAIMS.md` and the figure byte-identically (Step 4). Hand-typed items
are limited to dataset facts from dataset documentation, design constants in `PREREG.md` (grids, 3 seeds, 10,000 resamples,
frame sizes) and citation years.

Non-numeric claims checked:

| Claim in the paper | Status |
|---|---|
| H1 not supported, H2 supported | SUPPORTED (`results/tables/hypotheses.csv`; macros `hOneSupported`, `hTwoSupported`) |
| "No crossover": learned models at or above DTW from k=1 | SUPPORTED, point estimates 36/36; intervals exclude zero 36/36 (exploratory, dependent) |
| Euclidean 1-NN not worse than DTW at 50 signs | SUPPORTED (paired intervals) |
| Aspect-ratio correction avoids distorted angles ("the earlier implementation used normalised coordinates") | Code reading only; the effect on accuracy was NOT tested. Wording is a statement about the code, not a result |
| FastDTW slower "because of implementation" | WAS an unsupported interpretation; reworded to say the comparison reflects the implementations and equal-quality implementations were not compared |
| "a few seconds of training" | WAS inaccurate for the BiGRU; reworded to "seconds to a few minutes" with the numbers in Section 4.8 |
| Why DTW trails | Explicitly marked untested, no causal claim |
| Related-work characterisations | Each checked against the abstract or page (Step 5) |

No claim was left UNSUPPORTED after these edits.

---

## Step 3. Statistical audit

| Check | Result |
|---|---|
| Signer leakage | Signers disjoint across train/val/test (31/8/8), asserted in code and re-checked now |
| Duplicate content | 0 of 1,037 test and 0 of 1,007 val queries have an exact feature sequence in train; 0 shared file names |
| Test-set tuning | All choices on validation signers only (`results/tuned.json`, taken before the `prereg-v1` tag). Exception: ablation "best" values are maxima over many settings on the test signers; the paper says so |
| Standardisation leakage | Statistics from the selected references only, never val/test |
| Multiple comparisons | Two confirmatory hypotheses at 97.5%, H2 also at 99.17%; everything else labelled exploratory with no correction. The "36 of 36" count is of dependent comparisons on the same 8 signers |
| Missing CIs | Every table cell has a signer-cluster interval in `results/tables/main_top1.csv`; Table 1 and 2 in the paper show point estimates and refer to the CSV for intervals. A reviewer may want intervals in the paper |
| Inappropriate tests | No p-values; percentile cluster bootstrap with 8 clusters is coarse (few clusters), stated |
| Seeds | 3 seeds (draw and initialisation); the seed range is at most 3.4 points, against signer interval widths of 15-20 points |
| Cherry-picked examples | Per-sign lists are generated (worst signs, top confusion); no hand-picked examples |
| Causal wording | The discussion says explanations are untested |
| Researcher degrees of freedom | The design changed after validation dry runs (4 and 10 signs): per-family preprocessing, standardisation, step floor, DBA length. These are listed in `PREREG.md` Deviations and happened before any test scoring. Post-test additions (items 7-10) are listed too |
| Query sampling | Test queries are drawn once for the largest vocabulary and restricted for smaller ones; a standalone 10-sign run would draw different queries (documented) |

---

## Step 4. Reproduction test

Done in a fresh clone in a new environment (`uv venv --python 3.11`, `pip install -r requirements.txt`):

- `pytest`: 32 passed, 1 skipped (the real-data DTW-vs-tslearn test skips without the landmark cache).
- `python -m signstudy.report`: regenerated every table, the figure (PDF made deterministic), `paper/numbers.tex` and `CLAIMS.md`
  **byte-identically** to the committed files (`git diff --quiet` clean).
- `scripts/check_rerun.py` (needs the landmark cache): recomputed vocabulary-10 test rows for DTW 1-NN, Euclidean 1-NN and one CNN
  run (seed 0, k=5); ranks and predictions identical on all matched rows (3,552, 3,552 and 222).

What was found and fixed during the test: (1) the claims ledger embedded the git commit, so it could never match after a new commit;
(2) the report needed the multi-GB cache for the data statistics; now stored in `results/data_stats.json`; (3) the figure PDF
contained a timestamp; (4) the first version of the re-run check compared different query sets (a script flaw, not a result flaw).

What still breaks or was not tested: a full end-to-end `make quick` from nothing (hours of download and extraction) was not
re-run; the GitHub Actions workflow has not run on GitHub; learned-model determinism was shown on this Apple GPU only and may differ
on other hardware; extraction needs a network connection to the dataset server and its tar paths may change.

---

## Step 5. Citation audit

Every reference was fetched from Crossref or the arXiv API (`paper/verify/citation_lookup.txt`, `bib_log.txt`). The audit caught
three wrong DOIs that were guessed earlier (Tran et al., Saggio et al., Arcanjo et al.); the wrong ones never entered the bibliography.
Characterisations in the text were checked against abstracts or pages:

| Cited work | What the paper says | Checked |
|---|---|---|
| Bagnall & Lines 2014 | tuned NN-DTW is a strong baseline | Abstract: DTW classifiers far more competitive than Euclidean, esp. with tuned window |
| Ismail Fawaz et al. 2019 | deep networks competitive on large benchmarks | Abstract: competitive on UCR/UEA and multivariate sets |
| Arcanjo et al. 2024 | FastDTW on angular features of MediaPipe landmarks | Abstract page; also states it is competitive without deep learning |
| Saggio et al. 2020 | k-NN with DTW vs CNN on sensor-glove data | Abstract: ten Italian Sign Language gestures, seven participants; 96.6% vs 98.0% |
| Vandendriessche et al. 2025 | one-shot recognition with learned embeddings and a dictionary | Abstract |
| Jiang et al. 2021 | skeleton graph convolutional network | Abstract (SL-GCN) |
| Desai et al. 2023 | 2,731 signs, 52 signers, consented | Abstract |
| Desai et al. 2024 | hearing-researcher priorities, communication-barrier framing, unrepresentative data | Abstract |
| Mankar et al. 2024 | DTW vs LSTM, small custom dataset, no signer-independent evaluation | The paper in the earlier repository (custom dataset, 3 test cases; no signer split reported) |

Hedges: the paper uses "we did not find prior work that..." with the note that the search was not exhaustive; no "first" or "novel"
claim appears. Excluded because they could not be opened: the Atlantic article on sign-language gloves; the WFD statement.

---

## Step 6. Ethics, licensing and anonymity

- **Data licences:** PopSign v1.0 CC BY 4.0 (derivatives allowed). ASL Citizen was not used (its licence forbids redistribution).
  Released landmarks: hands and pose only; face landmarks and videos are never released.
- **Software licences:** MediaPipe Apache-2.0, tslearn BSD-2, fastdtw MIT; study code MIT (proposed, author to confirm).
- **Sensitive content:** videos show faces; not released. The dataset card lists identification and appearance-based use as out of
  scope; the study does neither. Offensive-content screening was not done (checklist B4: "partly").
- **Human subjects:** no new participants; no ethics approval needed beyond the original dataset's consent process.
- **Anonymity:** the draft is not anonymous. If the target venue is double-blind (WSLP at EMNLP was), these identify the author:
  the author line, `LICENSE`, `CITATION.cff`, the git author name and email in every commit, the GitHub user in any URL, the
  earlier-work self-citation (`mankar2024`, which names the same author), the PDF metadata, and `HUMAN_TODO.md`. Produce an
  anonymous mirror with rewritten history and a blinded citation before submitting there.
- **AI-use disclosure:** `LLM_DISCLOSURE.md` and the Reproducibility section answer ARR checklist E1; adapt to the final venue.

---

## Step 7. Rebuttal prep

1. **"Is DTW being handicapped?"** Answer: each family chose its preprocessing on validation signers (24 DTW options, 12 for
   models); zero-fill cost DTW 11.4 points on validation; richer features and dropped frames raised DTW to at best 71% against 93%.
   Pre-empt in a week: re-tune DTW at F4 and at each k; add DTW with per-dimension weights learned on the training set; k>1 NN;
   ROCKET.
2. **"Eight test signers."** Answer: true; intervals reflect it; the direction is the same in all 36 comparisons, but they are
   dependent. Pre-empt: run the same grid on the validation signers as a second, labelled-exploratory sample and on the ASL Citizen
   unseen-signer split (read the licence first; keep landmarks private).
3. **"Models are not converged / tuned at one condition."** Answer: grid edge and step cap acknowledged. Pre-empt: extend the epoch
   grid and cap, retune at k=1 and k=all on validation, report whether any number moves by more than the seed range (about 3 points).
4. **"Only two vocabulary sizes."** Answer: extraction of 50 signs took 8.8 hours; 100 and 250 were not run. Pre-empt: let the
   extractor run another 50 signs (about 9-10 hours) and rerun the grid, then 100.
5. **"Why does DTW trail?"** Answer: untested. Pre-empt: error analysis by sign type, examine alignment paths, test location-free vs
   location-aware features per sign, check whether the mirror/left-right labelling explains the dominant-hand effect (the
   dominant-slot ablation shows a large change for models, which suggests left/right slot assignment matters).

Additional one-week experiments: DTW with the band chosen per k; Mahalanobis or learned-metric DTW; low-power timing if a Raspberry
Pi class device can be borrowed; leave-one-signer-out on the training signers to widen the signer sample.

---

## Go / fix-first / do-not-submit: **FIX-FIRST**

Prioritised fix list:
1. (blocking) Fill author, affiliation, ORCID, acknowledgements; decide anonymity needs for the chosen venue.
2. (blocking) Do the Deaf-community feedback step, or state that it was not done.
3. Run a second dataset or at least a validation-signers replication so the paper is not single-dataset (ASL Citizen needs the
   licence read and private handling).
4. Extend vocabularies (run extraction to 100, ideally 250) or retitle and rewrite the claim as a 10-50 sign study.
5. Re-tune per k, extend the learned-model epoch grid and cap, and add one or two stronger baselines (ROCKET, k>1, per-dimension
   weighting).
6. Add an analysis of why (error analysis by sign type; mirror/handedness check).
7. Put intervals into Tables 1 and 2 in the paper.
8. Re-verify the venue's real call, template, page limit and policies; run the CI on GitHub; time on a low-power device if
   available.
