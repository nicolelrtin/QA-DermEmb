# QA-DermEmb literature evidence: release spec (contract between Claude Science and Claude Code)

**Version 1, 2026-11-26.** Both agents follow this file. Changing it takes a decision recorded in `DECISIONS.md` (Claude Code project) and in `methods_log.md` (Claude Science).

## Roles
- **Claude Science produces evidence.** It reads papers, extracts, validates, and writes **releases**. It does not aggregate across papers, rank, grade evidence, build profiles or make visualizations.
- **Claude Code consumes releases.** It does all analysis: evidence grading, subtype normalization, profiles, heatmaps and the website. It never edits a release. When it needs something from the literature, it files a **request**.

## Where things live: one small "exchange" git repo
Releases and requests travel through a **separate private GitHub repo**, `qa-dermemb-lit-exchange`. The main analysis repo never tracks them.

| | Mac (Claude Science side) | Server (Claude Code side, via the VS Code tunnel) |
|---|---|---|
| exchange clone | `~/Documents/A - Programming/QA-DermEmb_Literature/exchange/` | `projects/qa-dermemb-lit/exchange/` (**gitignored** by the main repo) |
| `releases/lit_vYYYYMMDD/` | written by Science | read-only for Code |
| `requests/NNN-*.md` | read by Science | written by Code |
| `requests/NNN-*.response.md` | written by Science | read by Code |

Both sides sync with `exchange.sh` (git pull, commit, push). Only the evidence releases and requests cross over; Science's working folder never does. Code commits a copy of each release's `manifest.json` into the main repo (`data/release_manifests/`), so analyses stay reproducible.

## Release folder contents (`lit_vYYYYMMDD/`)
| file | content |
|---|---|
| `papers.csv` | One row per paper (all 607 + any new). Columns: paper_id, crosswalk ids, title, year, journal, pub_types, priority_tier, text_status, reading_status, language, n_units, identity_check, notes. |
| `units.csv` | One row per lesion/lesion group (scope §3 columns + `unit_kind`, see request 001). |
| `unit_answers.parquet` | One row per unit × question (scope §7 2a). |
| `feature_frequencies.csv` | Group frequency rows (scope §7 2c) + request-001 columns. |
| `text_links.csv` | Feature → class links (scope §7 2b) + request-001 columns. |
| `descriptor_mentions.parquet` | Reading 1 descriptors. |
| `questions.csv` | N-500 + B list. Columns: question_id, source (N500 / B_main / B_single), question, topic, answer_source (image/text), coverage relation to N-500, n_papers. |
| `class_map.csv` | The 10 classes plus OTHER/UNRESOLVED, with short codes. |
| `data_dictionary.md` | Every column of every table: type, allowed values, meaning. |
| `validation_report.md` | V1–V9 results for this release. |
| `methods_log.md` | A copy of the log at release time. |
| `manifest.json` | Release id, date, scope_version, prompt versions, per-file row counts and SHA-256, `changes_since_previous`, `requests_resolved` (ids). |

## Rules
1. **A release is immutable.** Any change means a new `lit_vYYYYMMDD` (add a suffix `b`, `c` for same-day releases).
2. **Columns are added, never renamed or removed,** within schema major version 1. A breaking change bumps `schema_major` in the manifest and requires a request.
3. **Rows that fail validation stay in, flagged** (`validation_flags`), never deleted. Exact duplicate evidence rows are removed **before** release.
4. **No derived cross-paper numbers in a release.** No pooled frequencies, ranks, tiers, grades or profiles.
5. Every evidence row carries provenance (run_id, reader, model, prompt_version, scope_version, extracted_at).

## Requests
- **File name:** `NNN-short-title.md` (NNN zero-padded, increasing).
- **Header lines:** `status: open | answered | closed`, `priority`, `needed_for`.
- **Body:** what is needed, why, the acceptance check.
- **Science** answers with `NNN-short-title.response.md`: what it did, which release contains it, and anything it could not do. It lists the id in the next release's `manifest.json → requests_resolved`.
- **Code** verifies the acceptance check against the release, then sets `status: closed`.
