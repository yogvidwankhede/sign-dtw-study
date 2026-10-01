# Data, licences and what is released

## Used

| Dataset | Role | Licence / terms (as read on the official pages) | Consent |
|---|---|---|---|
| PopSign ASL v1.0, `game` category (Starner et al., NeurIPS 2023 D&B; Georgia Tech and DPAN) | primary | CC BY 4.0; derivatives allowed with attribution to Georgia Tech and DPAN. Source: signdata.cc.gatech.edu | Signers signed a consent form acknowledging a dataset intended for public use; Deaf adult signers; anonymised ids |
| ASL Citizen (Desai et al., NeurIPS 2023 D&B; Microsoft) | planned private replication | Non-commercial research only; you may not distribute the data or your modifications; use must match the consent; personal data must be destroyed when the research ends. Source: Microsoft "Dataset License" page | IRB approved, consented, Deaf team members |

The ASL Citizen licence text above is paraphrased from a summary of the licence page. Read the original before use.

## Handling

- Videos are downloaded one sign at a time, converted to landmarks and deleted. No video is stored in this repository or released.
- Face landmarks are a local cache for one ablation. They are never released, for either dataset.
- The ASL Citizen replication, if run, keeps its landmarks private and deletes its raw data at the end.
- The dataset cards ask not to use PopSign or ASL Citizen for identification or for applications that exploit participant
  identity or appearance, and not for continuous-signing or translation claims. This study does neither.

## Released from this study (once the author chooses to publish)

- Hand and upper-body pose landmarks derived from PopSign (CC BY 4.0, attribution to Georgia Tech and DPAN), produced by
  `scripts/package_landmarks.py`, together with extraction logs and versions.
- Code (MIT), configuration, per-query result tables, figures.

## Models and software

MediaPipe 0.10.14 (Apache-2.0), tslearn 0.9.0 (BSD-2-Clause), fastdtw 0.3.4 (MIT), PyTorch (BSD-style), NumPy/SciPy/pandas
(BSD). Landmark models change between MediaPipe versions, so extraction is pinned and every output records its versions.
