# When you decide to publish (nothing below has been done)

1. Create the repository (private first). Suggested description: "Pre-registered comparison of DTW and learned models for isolated sign recognition from hand landmarks (PopSign, ASL Citizen)".
   Suggested topics: `sign-language`, `dynamic-time-warping`, `mediapipe`, `pre-registration`, `reproducibility`, `accessibility`, `time-series-classification`.
2. Before pushing: confirm no ASL Citizen raw data is in the tree (`git ls-files | grep -iE "aslc|asl_citizen"` should list only aggregate files under
   `results_aslc/`, `figures_aslc/`, `paper/numbers_aslc.tex`, `CLAIMS_aslc.md`, `PREREG*`), and that `git log` does not contain cache files (history was cleaned on 2026-10-01).
   If the venue is double-blind, push the anonymous mirror (`python scripts/make_anon.py`) instead of this history.
3. Settings: enable "Private vulnerability reporting"; protect `main` (require CI); keep Actions on. The CI job regenerates the PopSign tables and fails on any difference.
4. Zenodo: link the GitHub account at zenodo.org, switch the repository on, then create a release `v0.1.0` from `release/RELEASE_NOTES_v0.1.0.md`. Add the DOI to `CITATION.cff` and the paper.
5. Optional: release hand and pose landmarks for PopSign only with `python scripts/package_landmarks.py OUT_DIR` (CC BY 4.0, attribute Georgia Tech and DPAN).
