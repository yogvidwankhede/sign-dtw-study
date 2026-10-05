# What the human author must do

Nothing here can be done by the code or by an assistant. Items marked (blocking) must be done before submission.

## Identity and metadata
- (blocking) Decide the author line and affiliation (student at WashU, or independent researcher), email, and ORCID. Fill them
  in `CITATION.cff`, `paper/paper.tex` and the submission form. No ORCID is on the Springer record for the earlier paper either.
- Add funding and acknowledgements (or state "none"). Acknowledge Deaf reviewers only with their consent.

## Licences and data
- (blocking for the replication) Read the ASL Citizen Dataset License and Recommended Use pages yourself (this study only had
  summaries), and email ASL_Citizen@microsoft.com if you want to release anything derived. Delete the ASL Citizen data when done.
- Confirm the code licence (MIT proposed) and that releasing hand and pose landmarks under CC BY 4.0 with attribution to
  Georgia Tech and DPAN is what you want. Then run `scripts/package_landmarks.py`.

## Publishing (nothing is pushed or public yet)
- Create the GitHub repository and decide when to make it public; push the branch.
- Mint a Zenodo DOI for the code and landmark release (link Zenodo to GitHub from your account) and add it to `CITATION.cff`.
- arXiv: first-time submitters need an endorsement; choose cs.CV, cs.CL or cs.LG as primary. Check the chosen venue's
  preprint policy first.

## Venue (verified 2026-09-30; re-check, nothing is open yet)
- No suitable venue has an open deadline. Watch the sign-language workshop at LREC (signlang, Hamburg site) and WSLP at
  ACL-family conferences after the 2027 workshop notifications (2 Oct 2026). Re-read the call for page limit, template,
  blind/anonymity rules, arXiv policy and the AI-disclosure policy, then swap the template in `paper/`.
- Fill in the venue's reproducibility/ethics checklist from `VENUE.md`.

## Community feedback
- Do the Deaf community feedback step in `ETHICS.md` before submitting, or state in the paper that it was not done.

## Earlier paper
- The repository for the earlier Springer paper reuses code from an MIT-licensed upstream that the paper does not cite.
  Decide with your co-authors whether to contact Springer about a correction. The repo-side attribution fix is on the
  `attribution-fix` branch of Sign_Language_Interpreter.
- Edit your Google Scholar and ORCID profiles yourself (steps were given in the chat summary).

## Added after the experiments (see `REVIEW.md` for the full audit and fix list)
- (blocking if the venue is double-blind) Produce an anonymous mirror: the git history contains your name and email; the paper,
  `LICENSE`, `CITATION.cff` and the self-citation identify you. Do not push this history to a public or reviewer-visible place.
- Decide whether to extend the study before submitting: vocabularies 100-250 (the extractor can continue: `make extract VOCAB=100`,
  about 9-10 hours per 50 signs on this machine), a second dataset, and the stronger baselines listed in `REVIEW.md`.
- Re-run `python scripts/check_rerun.py` and `python -m signstudy.report` after any change; CI will fail on a mismatch once the repo is on GitHub.
- The final paper text must be reviewed by you sentence by sentence; every number is generated, but the interpretations are drafts.
- If you push, the repository is about 2 MB (cache files were removed from history); the landmark cache (`data/landmarks`, about 3 GB)
  stays local. Release hand and pose landmarks with `scripts/package_landmarks.py` only after confirming the CC BY 4.0 attribution text.

## Added 2026-10-04 (second dataset and extensions)
- (licence) **Delete the ASL Citizen landmarks when the research is complete**: `rm -rf data/landmarks_aslc data/aslc data/tmp_aslc` and
  `results_aslc/raw` (per-query rows). The videos were already deleted after extraction. The licence requires destroying personal data and copies
  at the end of the research; decide when that is (after the final submission and any revision). Only aggregate ASL Citizen results are in git.
- Read the ASL Citizen licence (`use.txt` inside the dataset zip, also on the Microsoft page) yourself and confirm the use is consistent with it.
- The PopSign vocabularies 100 and 250 were not run (extraction is slow: about 10-15 minutes per sign on this machine); `make extract VOCAB=100` continues if you want them.
- The review (`REVIEW.md`) has a status table at the top showing which items from the first audit were addressed.

## Added 2026-10-05
- Submission materials now exist locally; see `SUBMISSION.md` for the table of what is ready and the short list of what only you can do.
- To send the community-feedback request: personalise `outreach/summary.tex` (name, contact, payment offer), compile it, and follow the steps in `ETHICS.md`.
- For a double-blind venue: run `python scripts/make_anon.py`, re-check the grep report is 0, and submit only that copy's PDF and repository.
