# Submission package: what exists and what is left

Everything below is local. Nothing has been pushed, uploaded or sent.

| Item | Where | State |
|---|---|---|
| Paper (ACL template, final mode) | `paper/paper.pdf`, sources in `paper/` | compiles; author/affiliation/ORCID placeholders |
| arXiv upload folder and zip | `python scripts/make_arxiv.py` -> `arxiv/`, `arxiv_upload.zip` | built and checked (all 29 references in the .bbl); metadata in `arxiv/METADATA.txt` |
| Anonymous version for double-blind review | `python scripts/make_anon.py` -> `../sign-dtw-study-anon` | 0 identifying matches, placeholder git identity, tests pass, paper compiles in review mode |
| Community feedback summary (2 pages) | `outreach/summary.tex`, `outreach/summary.pdf` | ready to personalise and send; the outreach itself is yours |
| Venue checklist, AI-use disclosure | `CHECKLIST.md`, `LLM_DISCLOSURE.md` | filled in |
| Zenodo record, release notes, GitHub setup steps | `.zenodo.json`, `release/` | drafts; creators are placeholders |
| Draft PR descriptions for Sign_Language_Interpreter | `release/sign_language_interpreter_prs.md` | drafts; PRs not opened |
| Pre-registration and addendum, claims ledger, audit | `PREREG.md`, `PREREG_ADDENDUM.md`, `CLAIMS.md`, `REVIEW.md` | final |
| Reproduction | `make setup`, `make test`, `python -m signstudy.report`, CI workflow | clean-clone check passes; GitHub run not done |

## Only you can do
1. Author line, affiliation, ORCID, funding and acknowledgements (paper, `CITATION.cff`, `.zenodo.json`).
2. Send the community feedback summary (and pay reviewers); record what changed, or state in the paper that no feedback was obtained.
3. Decide the venue once a call opens; re-read its template, page limit, blind rules, arXiv policy and AI-disclosure policy. Use the anonymous mirror if it is double-blind.
4. Publish: create the GitHub repository, push, enable settings, link Zenodo (`release/GITHUB_SETUP.md`); submit to arXiv (needs an endorsement).
5. Delete the ASL Citizen landmarks when the research is complete, and read its licence yourself (`HUMAN_TODO.md`).
