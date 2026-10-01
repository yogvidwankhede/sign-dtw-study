# Venue notes (looked up 2026-09-30; none of these has an open deadline today)

Recommendation: post to arXiv first, then target the next sign-language workshop call. All dates below must be re-checked.

| Rank | Venue | What was verified | Not verified |
|---|---|---|---|
| Primary | Workshop on the Representation and Processing of Sign Languages (signlang), LREC series, Univ. Hamburg site (2026 edition: Palma, 16 May 2026, closed) | 4-8 pages, single-blind (authors identified), LREC style; no arXiv conflict from blind rules | 2027 edition, dates, arXiv and AI-use policy |
| Backup 1 | Workshop on Sign Language Processing (WSLP), EMNLP 2026 (closed 24 Aug 2026), watch for the next edition | 8 / 4 page limits, ACL template, double-blind via OpenReview, dual submission allowed if the other venue allows it | arXiv, LLM and checklist policy; a 2027 edition |
| Backup 2 | ASSETS 2027 poster / late-breaking | ASSETS 2026 is 25-28 Oct 2026 (Porto) | 2027 call, page limit, template, AI policy; weaker fit unless the paper adds a Deaf-user angle |

ACL-family rolling-review venues require (per the ARR call): a Limitations section, the Responsible NLP checklist,
disclosure of generative-AI use, anonymous submission, and no simultaneous submission elsewhere. Check the anonymity-period
rules before posting a preprint to an ARR-routed workshop.

arXiv: first-time submitters need an endorsement.

## Checklist items to prepare (typical)
- Limitations: `paper/` section "Limitations". Ethics: `ETHICS.md`. Data licences: `DATA.md`. Code and data release: `README.md`.
- Reproducibility: `Makefile`, pinned `requirements.txt`, seeds in `signstudy/config.py`, per-run provenance JSON next to every
  result file, `CLAIMS.md`.
- Compute: Apple M4 Max, no cloud spend; extraction time recorded in `data/landmarks/EXTRACTION_LOG.jsonl`.
- AI-use disclosure: `LLM_DISCLOSURE.md`.
