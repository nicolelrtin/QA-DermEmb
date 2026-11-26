# QA-DermEmb literature: plan

Status key: [ ] todo · [~] in progress · [x] done · [!] blocked (say on what)

## M0: Set up (Claude Code)
- [ ] Create the project skeleton:
  - `src/`, `tests/`, `mappings/`, `data/release_manifests/`;
  - `config.yaml`, a Makefile with `make profiles`, `make site`, `make test`.
- [ ] Add the `.gitignore` entry `projects/qa-dermemb-lit/exchange/`. The user clones the private exchange repo there (it already holds `requests/001-evidence-fields.md`).

## M1: Interim release (lit_v0, the current tables, before request 001)
- [ ] On the Mac, the user runs `exchange/exchange.sh --v0`, which packages the current Science tables as `lit_v0`. Then `git pull` in `exchange/` on the server.
- [ ] Prototype grading and the page on lit_v0, with fallbacks:
  - `cited_claim` from the note text;
  - `has_counts` from freq_k/freq_n;
  - subtype = `unspecified`;
  - exact-duplicate rows dropped in code.

  Mark everything **provisional**.

## M2: Release v1 (after Science answers request 001)
- [ ] Pull `lit_v1` in `exchange/`, pin it, run the schema and acceptance tests, close request 001.
- [ ] Subtype mapping `mappings/subtypes.csv`: draft, then user review.
- [ ] Graded profiles per class and per subtype, with counts verified.

## M3: Website page
- [ ] **Per-class dropdowns.** Each shows the shortlist, then the buffer collapsed below it. Each feature shows:
  - grade counts (strong / supporting / contrary papers);
  - pooled frequency with n and number of papers;
  - subtype tags and match level.

  Expanding a feature shows its evidence lines: paper, quote, location, k/n, link strength, histopathology confirmation.
- [ ] Threshold controls (min_n, strong_freq, weak_freq) and per-class counts.
- [ ] Checkboxes per feature, plus an "Export selected" CSV for the dermatologist.
- [ ] **Class × class heatmap** of shared shortlist features (count and Jaccard). Click a cell to list the shared features.
- [ ] **UpSet plot** across the 10 classes.
- [ ] **Venns** only for MEL / NEVUS / INDET_MEL and BCC / SCC / INDET_EPI.
- [ ] **Caveats panel:**
  - text coverage (354/607 papers at lit_v0);
  - histopathology confirmation is partial;
  - INFLAM and VASC are umbrella classes (show their subtypes).

## M4: Dermatologist round
- [ ] Export the shortlist; record her agreement as `mappings/derm_review_YYYYMMDD.csv`; summarize in FINDINGS.

## M5: Side project, literature features vs the 500 questions
- [ ] Use `questions.csv` coverage (same / narrower / broader / none) plus the graded profiles: which supported literature features have no N-500 question, and which N-500 questions have no literature support.

## Later: weekly literature pipeline (HANDOFF v2 Part B)
On hold until M4. When it resumes, new papers arrive as Science releases; the collector code goes under `projects/qa-dermemb-lit/collector/`.
