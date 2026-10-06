# Venue checklist (ACL Responsible NLP Research checklist, fetched 2026-09-30; adapt if the final venue differs)

Answers refer to the paper and repository. "Pending" items need the author or the final results.

## A. Every submission
- A1 Limitations described: yes (paper, "Limitations").
- A2 Risks discussed: yes (paper, "Ethics and broader impact"; `ETHICS.md`).

## B. Scientific artifacts
- B1 Creators cited: yes (PopSign, MediaPipe, tslearn, fastdtw, earlier implementation).
- B2 Licence and terms discussed: yes (`DATA.md`: PopSign CC BY 4.0; ASL Citizen Microsoft Research License, used under its terms with aggregate-only release and deletion at the end; code MIT proposed).
- B3 Use consistent with intended purpose: yes. PopSign's card lists isolated sign recognition as suitable and continuous signing,
  translation and identification as unsuitable; the study does only the former.
- B4 Checked for identifying or offensive content: partly. Signer ids are anonymised by the dataset; videos show faces and are not
  released; face landmarks are kept local; the data were not screened for offensive content.
- B5 Documentation: yes (`DATA.md`, `README.md`, `PREREG.md`).
- B6 Statistics of data and splits: yes (paper, Data and methods; `results/tables/data_stats.csv`).

## C. Computational experiments
- C1 Parameters, compute, infrastructure: yes. About 0.46-0.57 M parameters per model (the embedding baseline is the same CNN); Apple M4 Max laptop, no cloud; extraction and
  training hours are generated macros in the paper's reproducibility text (pending: final wording).
- C2 Setup and hyperparameter search: yes (tuning grids in `results/tuned.json` and `results/tables/tuning_validation.csv`).
- C3 Descriptive statistics: yes (signer-clustered intervals; seed spread in `results/raw`).
- C4 Packages and settings: yes (`requirements.txt`, provenance JSON beside every result).

## D. Human participants
- Not applicable: no new participants. The study uses an existing consented dataset. The Deaf-community feedback step in
  `ETHICS.md` is pending and, if not done, must be stated as not done.

## E. AI assistants
- E1 Use disclosed: yes (`LLM_DISCLOSURE.md`; paper, Reproducibility).
