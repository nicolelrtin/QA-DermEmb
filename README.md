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
