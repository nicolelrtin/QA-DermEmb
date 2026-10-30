# qa-dermemb

Dermatology question generation and analysis. Poetry environment `qa-dermemb` (Python 3.11).

| Folder | What it does |
|---|---|
| `2026-10-08 Generated Questions/` | Four CSV question sets (500 each), generated with different methods |
| `Question Embeddings/` | Embeds all four sets with three models and plots coverage with UMAP |
| `Image Analysis/` | PanDerm image embeddings, zero-shot diagnosis, question relevance per image |
| `dermlip.py` | Shared DermLIP-PanDerm loader used by both |

## Models

| Name | Hugging Face repo | Notes |
|---|---|---|
| `qwen3` | `Qwen/Qwen3-Embedding-8B` | General-purpose SOTA text embedder (`--qwen3-size 0.6B/4B` for smaller) |
| `panderm` | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | PanDerm-base vision + PubMedBERT text, CLIP-aligned. **Gated**: click "Request access" on the model page (auto-approved). Needs the Derm1M open_clip fork |
| `skingpt` | `yuhos16/SkinGPT-R1` | Qwen2.5-VL-7B + dermatology LoRA, ~17 GB, needs ~18 GB GPU memory. Text embeddings = mean-pooled last hidden layer |

PanDerm itself is image-only; its "text embedding" here is DermLIP's text tower, trained to share PanDerm's image space. The original SkinGPT-4 is not on Hugging Face and its LLM (Llama-2-13b-chat) was frozen, so SkinGPT-R1 is used instead.

## Setup on the Linux workstation

```bash
cd "<project folder>"
poetry install                # creates ./.venv, prompt (qa-dermemb)
./setup_derm1m.sh             # clones the Derm1M fork into third_party/ (pinned commit)
hf auth login                 # or export HF_TOKEN
poetry shell
```

## Question coverage (UMAP)

```bash
python "Question Embeddings/embed_questions.py"     # ~25 GB of downloads on first run
python "Question Embeddings/plot_umap.py"
```

Outputs in `Question Embeddings/outputs/`:
- `umap_<model>_by_set.png`: one panel per set, highlighted over the others in gray (best view of coverage)
- `umap_<model>.png`, `umap_all.png`: all sets overlaid, per model and side by side
- `umap_coords.csv`: every point with its question text, for looking up clusters

## Image analysis (PanDerm)

```bash
python "Image Analysis/panderm_image_analysis.py" path/to/images \
    --labels melanoma nevus "basal cell carcinoma" "seborrheic keratosis" \
    --questions "2026-10-08 Generated Questions/2026-10-13_binary_dermal_image_questions_A.csv"
```

Writes `image_embeddings.npy`, `zero_shot.csv` and `question_relevance.csv` (top-10 most relevant questions per image; relevance, not answers) to `Image Analysis/outputs/`.

DermLIP weights are CC BY-NC-ND 4.0: non-commercial research only, no redistribution.

## QA-Emb pipeline (`qaemb/`)

Interpretable dermoscopy embeddings: Qwen2.5-VL answers yes/no dermatology questions per image (and per
grid patch), P(yes) from the Yes/No logits becomes the feature vector, and linear/MIL classifiers predict
malignant vs benign. Plan and decisions: `handoff/HANDOFF_QA-Emb-Derm.md`.

```bash
python -m qaemb.run stage=<manifest|prepare|answer|embed|model|all> +exp=[<ID>,...] tag=<text> [key=value ...]
python -m qaemb.run stage=all +exp=[...] dry_run=true          # resolved config + cache hits only
python -m qaemb.report modes <answer_dir> <answer_dir> ...      # timing + agreement between answer modes
python -m qaemb.report cost <answer_dir> --images 1000 --G 4 --questions 1000 --local 466
```

- **Config**: `conf/base.yaml`, experiments in `conf/exp/<group>/<ID>.yaml` (one per group; two IDs from the
  same group are rejected). Any key can be overridden as `key=value`.
- **Caching**: each stage writes to `outputs/qaemb/<stage>/<hash>/`, hash = stage config + upstream hashes +
  stage `VERSION`. Finished stages are skipped; bump `VERSION` in a stage module when its code changes output.
- **Labels**: `label_2c` (malignant 1, benign 0). `label.indeterminate`: `holdout` (default; label -1, answered
  and scored but never fitted or in metrics), `drop`, `benign`, `malignant`.
- **Splits**: grouped by `lesion_id` (no patient ID available), stratified 70/15/15; `D0_smoke10` uses leave-one-out.
- **Questions**: `2026-10-15 Generated Questions/DermQs_Source-1015FT_Prompt-C_N-500_Negated.csv`. Each
  Question ID appears twice: `original = 1` (as written) and `original = 0` (its negated complement). Drafted
  `scope` (local/global/metadata) and `kind` (clinical/artefact) live in the sidecar `..._annotations.csv`,
  keyed by Question ID. The question-set hash covers (ID, original, text) only.
- **Answer modes** (`mode` group): `M_stateless` (one question per sequence), `M_topic` (one prompt per
  original x topic), `M_all` (one prompt per original value). Originals and complements never share a prompt.
- **Adjusted answer**: `p_adj = P(yes | original) - P(yes | complement)`, in [-1, 1]; near 0 means the model
  gave the same answer to a question and its negation. It is the default feature (`F_adj`; also `F_adj_bin`,
  `F_orig`, `F_orig_bin`). `incoherence = P(yes|orig) + P(yes|compl) - 1` (< 0: "no" to both) and the
  contradiction rate are reported at S4.
- **Answerers** (`answerer` group): `ANS_qwen7b` (Qwen2.5-VL-7B, P(yes) from logits), `ANS_skingpt_r1` (SkinGPT-R1;
  its reasoning is skipped by starting its reply with `<think></think><answer> Final answer:`, so its next word is
  the answer), `ANS_claude_opus` (Claude Opus 5.5 via the Messages/Batch API, effort low: *stated* P(yes) and
  confidence as forced JSON; needs an API key in `ANTHROPIC_API_KEY`; `claude.use_batch=false` makes direct calls)
  and `ANS_claude_opus_cli` (the same prompts through `claude -p` on a Claude subscription, no API key). Claude is
  not deterministic, so replicates are separate runs: `rep` group `R1`–`R4`. A Claude prompt that does not return
  a numeric answer for every question is retried (`claude.max_attempts`); if it still fails, the run stops.
- **Long Claude runs on the subscription** go through a queue that cron resumes after usage-limit pauses: put one
  command per line in `outputs/qaemb/queue/queue.txt` (e.g. `qaemb.run stage=embed +exp=[…,ANS_claude_opus_cli,…] tag=…`);
  cron runs `python -m qaemb.queue run`. The runner pauses when 5-hour or weekly usage passes `claude_cli.stop_at`
  and continues after the reset; `python -m qaemb.queue status` shows progress. This usage is shared with your
  own Claude Code and claude.ai use.
- **P_w** = P(yes | original) × |p_adj| (feature source `F_w`); Claude's stated confidence is kept per version.
  `python -m qaemb.report replicates <run_id> …` aggregates replicates; `compare` accepts `A+B+C+D` to average them.
- **Tokens**: every run report has a Tokens section (per run, per image, per question) and writes
  `tokens_by_image.csv` / `tokens_by_question.csv` next to it.
- **Reports**: every run writes `outputs/qaemb/runs/<run_id>/report.md` (what ran, stage outputs, results,
  metrics, top questions, link to `answers_readable.csv`), linked from `experiments_log.md`. Each experiment
  batch gets a write-up in `reports/<date>_<name>/` (tracked in git); index in `reports/README.md`.
- **Run records**: `outputs/qaemb/runs/<run_id>/run_meta.json`, MLflow at `outputs/mlflow.db`
  (`mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`), one row per run in `experiments_log.md`.
