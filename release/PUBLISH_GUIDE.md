# Your click-by-click publish guide (nothing here has been done for you)

State on 2026-10-13: private repo https://github.com/yogvidwankhede/sign-dtw-study holds `study-setup` and tags
`prereg-v1`, `prereg-v2`. Sign_Language_Interpreter `main` was fast-forwarded locally (not pushed).

## A. Before anything public
1. Send the two feedback emails (`outreach/EMAIL_*.md`). Wait for replies (a week is reasonable), then record changes or
   state in the paper that feedback was requested and not received.
2. Read the ASL Citizen licence yourself. If you are done with the replication, delete local data (section D).
3. Re-read `paper/paper.pdf` end to end. Check every number you plan to quote.

## B. Make the repo public (only after A)
1. GitHub > repo > Settings > General > change visibility to Public. First confirm
   `git ls-files | grep -iE "aslc|asl_citizen"` lists only aggregate files.
2. Settings > Code security: enable private vulnerability reporting. Branch protection on `main`: require CI.
3. Merge `study-setup` into `main` through a pull request (the CI run must be green).

## C. Zenodo DOI
1. zenodo.org > log in with GitHub > Account > GitHub > switch the repo ON.
2. GitHub > Releases > new release `v0.1.0` (notes: `release/RELEASE_NOTES_v0.1.0.md`). Zenodo mints the DOI.
3. Put the DOI in `CITATION.cff` and the paper, then rebuild (`python -m signstudy.report`, `scripts/make_arxiv.py`).

## D. ASL Citizen data deletion (when the research is complete)
```
cd ~/sign-dtw-study
rm -rf data/landmarks_aslc data/aslc results_aslc/raw
```
`results_aslc/raw` is git-ignored; aggregate tables and figures stay. After this the ASL Citizen numbers can no longer be
recomputed from raw rows (the committed aggregates and the PREREG record remain).

## E. arXiv
1. arxiv.org > register > first-time submitters need an endorsement for the category (cs.CV or cs.LG): request via the
   submission page, ask a colleague who has posted there.
2. New submission > upload `arxiv_upload.zip` (rebuild with `python scripts/make_arxiv.py`); metadata is in `arxiv/METADATA.txt`.
3. Check the venue's preprint policy first. If the venue is double-blind, post only after checking its rules.

## F. Sign_Language_Interpreter (the earlier paper's repo)
`main` now contains the attribution fix, OSS standards and tests locally. To publish: `cd ~/Sign_Language_Interpreter &&
git push origin main` (this changes your public repo). PR text drafts: `release/sign_language_interpreter_prs.md`.
Decide with co-authors whether to ask Springer for a correction (it does not cite the upstream MIT repo).
