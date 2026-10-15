"""Shared loader for DermLIP-PanDerm (PanDerm-base vision tower + PubMedBERT text tower).

The model only loads through the open_clip fork in the Derm1M repo, which builds the
PanDerm CAE ViT. Run ./setup_derm1m.sh once to fetch it into third_party/.
Weights are gated: request access at
https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256 (auto-approved).
"""
import sys
from pathlib import Path

import torch

MODEL_ID = "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256"
DERM1M_SRC = Path(__file__).resolve().parent / "third_party" / "Derm1M" / "src"


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _import_fork():
    if not DERM1M_SRC.exists():
        raise SystemExit(f"Derm1M fork not found at {DERM1M_SRC}. Run ./setup_derm1m.sh first.")
    if str(DERM1M_SRC) not in sys.path:
        sys.path.insert(0, str(DERM1M_SRC))
    import open_clip

    if not Path(open_clip.__file__).resolve().is_relative_to(DERM1M_SRC):
        raise RuntimeError(
            f"Imported open_clip from {open_clip.__file__}, not the Derm1M fork. "
            "Uninstall open_clip_torch from this environment."
        )
    return open_clip


def load_dermlip(device: str):
    """Return (model, preprocess, tokenizer) with the model in eval mode on `device`."""
    open_clip = _import_fork()
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)

        # transformers 5 removed batch_encode_plus, which the Derm1M fork still calls.
    # Calling the tokenizer directly takes the same arguments and returns the same result.
    hf_tok = getattr(tokenizer, "tokenizer", None)
    if hf_tok is not None and not hasattr(hf_tok, "batch_encode_plus"):
        hf_tok.batch_encode_plus = hf_tok.__call__
        
    return model.to(device).eval(), preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: str, batch_size: int = 64) -> torch.Tensor:
    """L2-normalised text embeddings in the shared image-text space (N x 512)."""
    out = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i : i + batch_size]).to(device)
        out.append(torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    return torch.cat(out)
