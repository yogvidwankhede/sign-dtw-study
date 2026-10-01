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
