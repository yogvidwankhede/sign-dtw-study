# Draft pull-request descriptions for ~/Sign_Language_Interpreter (not opened; merge in this order)

## 1. attribution-fix -> main
**Add upstream attribution, MIT licence notice and honest scope notes.** This repository is derived from gabguerin/Sign-Language-Recognition--MediaPipe-DTW (MIT). Several files are identical to upstream and the
rest differ by docstrings and a few lines. The PR restores the upstream copyright and permission notice (`LICENSE`), documents what is upstream and what was added, adds `CITATION.cff`/`CITATION.bib` for the published paper
(doi 10.1007/978-981-97-1488-9_7, author order as in the Springer record), and states plainly that the "NLP" step is a keyword lookup and that the project recognises isolated signs only.

## 2. oss-standards -> attribution-fix (or main after 1)
**Community files, CI and a requirements fix.** Adds CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue and PR templates, Dependabot and a compile-check workflow; replaces the misspelled `tdqm` requirement by `tqdm`, adds the
missing `spacy` and `pyttsx3`, and removes an unparseable editor scratch file.

## 3. tests-and-audit -> oss-standards (or main after 2)
**Fix `dtw_distances` under current pandas, add tests, correct README.** `iterrows()` yields copies, so distances stayed 0 on pandas 3.0.6 (verified); results are collected and assigned back as a column. Adds six tests (3 of the 6 fail without the fix and all pass with it), a CI test step, and README corrections (missing-hand handling, voting defaults: code `batch_size=1`, paper 2, upstream 5).
