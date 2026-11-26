# QA-DermEmb literature: decisions log

Format: date · decision · rationale · who

- 2026-11-20 · Scope v2.1 frozen for reading (calibration rulings in scope §16) · calibration rounds 1–4 · user + Claude Science
- 2026-11-23 · B list v1: main list (≥ 2 papers, 363 questions) answered in Reading 2; single-paper list (395) not answered · keep Reading 2 tractable · user
- 2026-11-26 · Two-agent split: Claude Science produces evidence releases only; Claude Code does all analysis, planning (this folder) and the website · the profile analysis drifted inside Science, and three agents were too hard to coordinate · user
- 2026-11-26 · Claude Science's `canonical_profiles.csv` / `profiles_summary.md` are retired; profiles are rebuilt here from releases · it counted any frequency (even 0–2%) as support and pooled subtypes · user + review
- 2026-11-26 · Evidence grading strong / supporting / not supporting / contrary (see CLAUDE.md); **one strong paper is enough** for the shortlist; uniqueness is not a filter (shown in the heatmap instead) · a single frequency study or a stated tissue explanation can be decisive; descriptors vary across papers, so keep a visible buffer · user
- 2026-11-26 · Subtypes are normalized here from `diagnosis_as_reported` (request 001 adds it to every evidence row); site/subtype-specific features are labelled · MEL alone has 239 subtype wordings; acral/nail features were pooled into general MEL · user + review
- 2026-11-26 · Releases and requests travel through a separate private GitHub repo (`exchange/`, gitignored by this repo) · keep the main repo clean; the server is reached through a VS Code tunnel (no SSH), and git/GitHub works on both ends · user
