status: open
location: exchange/requests/ (this copy is the template)
priority: high
needed_for: M2 (graded profiles, subtype facets, histopath display)
sent: 2026-11-26 (delivered to Claude Science inside HANDOFF_v3.md)

# 001: evidence fields for grading and subtypes

Full text: HANDOFF_v3.md, section "Request 001". In short:
1. `diagnosis_as_reported` on every feature_frequencies and text_links row.
2. `unit_id` on feature_frequencies rows that come from a group unit.
3. `cited_claim` boolean on both tables.
4. `has_counts` on feature_frequencies; qualitative rows move to links or are dropped (logged).
5. De-duplicate exact duplicate evidence rows (about 263 frequency and 87 link rows in the current tables).
6. `unit_kind` (described_lesion / literature_summary) on units.csv.
7. DL016 set to wrong_document and its rows removed; record the language version read for LS131/LS138/DL359.

## Acceptance (run by tests/test_request_001.py)
- 100% non-empty `diagnosis_as_reported`; `cited_claim` ∈ {true, false} on all rows.
- `has_counts` present; no row with `has_counts = false` and frequency = 0.
- 0 exact duplicates on (paper_id, question_id, class, quote).
- `unit_kind` on all units.
- `manifest.requests_resolved` contains "001".
