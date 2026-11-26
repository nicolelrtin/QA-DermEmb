# QA-DermEmb literature: project rules (inherits root PROJECT.md / FINDINGS.md)

## Purpose
Turn the literature evidence produced by Claude Science into:
1. a short, well-supported feature list per lesion class, for a dermatologist to confirm;
2. views of how classes share features (heatmap, UpSet, selected Venns);
3. later, the overlap between literature features and the 500 image questions (side project; see PLAN).

## Data
- **Inputs:** releases in `exchange/releases/lit_vYYYYMMDD/`. `exchange/` is a separate private git repo, gitignored by this repo; sync it with `exchange/exchange.sh`. Copy each release's `manifest.json` to `data/release_manifests/` and commit it.
- **Read-only:** never modify a release. Pin the release id in `config.yaml` (`release: lit_vYYYYMMDD`). Outputs record the release id they were built from.
- **Schema guard:** `tests/test_release_schema.py` checks that the pinned release has every column the code uses, plus the manifest checksums. If it fails, stop and tell the user; do not patch around it.
- Feature ids: `DQ###` = the 500 questions, `B####` = literature (B list) questions. Pool a B question into its N-500 id only where `questions.csv` says relation = same.

## Evidence grading (the analysis rule; parameters live in `config.yaml`)
Unit of evidence: one (class × feature × paper) item, after dropping exact duplicates and rows that fail validation (V1–V6).

| grade | condition (defaults) |
|---|---|
| **Strong** | (a) frequency with counts: `has_counts`, n ≥ `min_n` (20), frequency ≥ `strong_freq` (0.25), match exact/close, not `cited_claim`; **or** (b) towards link with a stated tissue explanation: strength 1, mechanism not "association only", match exact/close, not `cited_claim` |
| **Supporting** | association-only towards links; frequencies without counts or with n < `min_n`; partial matches; cited claims; frequencies between `weak_freq` (0.10) and `strong_freq`; individually described cases (`unit_kind = described_lesion`, unit_type case) stating the feature present |
| **Not supporting** | frequency < `weak_freq` or 0; statements of absence; duplicates; failed validation |
| **Contrary** (counted separately) | away links (direction −) |

Feature level, per class:
- **Shortlist:** at least 1 strong paper. One strong study is enough; do not require multiple papers.
- **Buffer:** no strong paper, but supporting items from at least 2 distinct papers. Shown collapsed, not hidden.
- **Contra flag:** contrary papers ≥ strong + supporting papers.
- **Ranking within the shortlist:** number of strong papers, then pooled frequency of the strong frequency items (Σk/Σn), then number of supporting papers.
- **Not a filter:** uniqueness to a class. Overlap belongs in the heatmap.
- **Display attributes, not filters:** histopathology confirmation (via `unit_id`), subtype, match level.
- Every threshold is a page control and a `config.yaml` key. Report counts per class for the defaults.

## Subtypes
- Normalize `diagnosis_as_reported` into a controlled subtype list per class in `mappings/subtypes.csv` (raw string → class, subtype). Build it with code plus model help, then **have the user review it**. Unmapped strings stay `unspecified`.
- Profiles can be shown for the whole class and per subtype. Site- or subtype-specific features (acral, nail, facial, mucosal; Spitz) are labelled as such, so they don't read as general class features.
- `literature_summary` units do not count as lesions in Experiment A vectors. They may count as supporting evidence in profiles.

## Engineering
- Code lives in `src/`, tests in `tests/`, notebooks only for exploration (not imported).
- Analysis must be reproducible: `make profiles` rebuilds everything from the pinned release.
- **Every reported count must be checked** against an independent pandas computation in a test.
- Requests to Claude Science go in `exchange/requests/`, as RELEASE_SPEC describes. Write them, then run `exchange/exchange.sh`.
- Keep the project small: no copies of release data in other folders, and no generated files committed except the small summary tables used by the website.
