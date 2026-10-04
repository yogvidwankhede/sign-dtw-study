# Data, licences and what is released

## Used

| Dataset | Role | Licence / terms (as read on the official pages) | Consent |
|---|---|---|---|
| PopSign ASL v1.0, `game` category (Starner et al., NeurIPS 2023 D&B; Georgia Tech and DPAN) | primary | CC BY 4.0; derivatives allowed with attribution to Georgia Tech and DPAN. Source: signdata.cc.gatech.edu | Signers signed a consent form acknowledging a dataset intended for public use; Deaf adult signers; anonymised ids |
| ASL Citizen (Desai et al., NeurIPS 2023 D&B; Microsoft) | pre-registered replication (used) | Non-commercial research only; you may not distribute the data or your modifications; use must match the consent; personal data must be destroyed when the research ends. Source: Microsoft "Dataset License" page | IRB approved, consented, Deaf team members |

ASL Citizen's terms were read in full on 2026-10-04 from the `use.txt` file bundled with the dataset (Microsoft Research License): non-commercial,
non-revenue research only; no distribution of the data or modifications; use consistent with participants' consent; no use of personal data to contact
anyone; destroy personal data and copies when the research is complete. The author must still read the licence personally.

## Handling

- Videos are downloaded one sign at a time, converted to landmarks and deleted. No video is stored in this repository or released.
- Face landmarks are a local cache for one ablation. They are never released, for either dataset.
- ASL Citizen: only the needed videos were fetched by HTTP range request from the official download, converted to hand and pose landmarks (no face
  landmarks at all) and deleted. Landmarks live in the git-ignored `data/landmarks_aslc/` and must be deleted when the research is complete
  (`HUMAN_TODO.md`). Only aggregate results (accuracy tables, figures, macros) are committed; per-query rows and sign names stay local.
- The dataset cards ask not to use PopSign or ASL Citizen for identification or for applications that exploit participant
  identity or appearance, and not for continuous-signing or translation claims. This study does neither.

## Released from this study (once the author chooses to publish)

- Hand and upper-body pose landmarks derived from PopSign (CC BY 4.0, attribution to Georgia Tech and DPAN), produced by
  `scripts/package_landmarks.py`, together with extraction logs and versions.
- Code (MIT), configuration, per-query result tables, figures.

## Models and software

MediaPipe 0.10.14 (Apache-2.0), tslearn 0.9.0 (BSD-2-Clause), fastdtw 0.3.4 (MIT), PyTorch (BSD-style), NumPy/SciPy/pandas
(BSD). Landmark models change between MediaPipe versions, so extraction is pinned and every output records its versions.
