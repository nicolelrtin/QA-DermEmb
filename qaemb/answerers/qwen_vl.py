"""Qwen2.5-VL answerer: P(yes) from the Yes/No logits of the first answer token.

The image prefix (chat header + image tokens + fixed instruction) is prefilled once per unit and
its KV cache reused for every question. Continuations get explicit 3-D mRoPE position ids
(prefix_len + j + rope_delta), so a batch of independent question sequences can share one prefix.
"""
import copy
import time
from dataclasses import dataclass

import torch
from PIL import Image

SENTINEL = ""  # private-use char marking where the variable text starts


@dataclass
class Prefix:
    cache: object
    length: int       # tokens in the prefix
    next_pos: int     # mRoPE text position of the first token after the prefix
    ms: float


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class QwenVL:
    def __init__(self, acfg, revision: str, max_pixels: int, min_pixels: int):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = acfg.device
        self.dtype = acfg.dtype
        sub = {"subfolder": acfg.subfolder} if acfg.get("subfolder") else {}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            acfg.checkpoint, revision=revision, dtype=getattr(torch, acfg.dtype), attn_implementation=acfg.attn, **sub
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            acfg.checkpoint, revision=revision, size={"shortest_edge": min_pixels, "longest_edge": max_pixels}, **sub
        )
        self.tok = self.processor.tokenizer
        self.yes_ids = self._single_token_ids(acfg.yes_tokens)
        self.no_ids = self._single_token_ids(acfg.no_tokens)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def _single_token_ids(self, words) -> list[int]:
        ids = {t[0] for w in words if len(t := self.tok.encode(w, add_special_tokens=False)) == 1}
        if not ids:
            raise RuntimeError(f"None of {list(words)} is a single token")
        return sorted(ids)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def split_template(self, user_text: str) -> tuple[str, str]:
        """Chat-format user_text (which contains SENTINEL) and split it at the sentinel."""
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
        full = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pre, post = full.split(SENTINEL)
        return pre, post

    @torch.no_grad()
    def prefill(self, image: Image.Image, prefix_text: str) -> Prefix:
        _sync()
        t = time.perf_counter()
        inputs = self.processor(text=[prefix_text], images=[image], return_tensors="pt").to(self.device)
        out = self.model(**inputs, use_cache=True)
        _sync()
        L0 = inputs["input_ids"].shape[1]
        delta = int(self.model.model.rope_deltas.reshape(-1)[0])
        return Prefix(out.past_key_values, L0, L0 + delta, (time.perf_counter() - t) * 1e3)

    def p_yes(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(P(yes | yes or no), P(yes or no)) from last-position logits [..., V]."""
        logp = torch.log_softmax(logits.float(), dim=-1)
        ly = torch.logsumexp(logp[..., self.yes_ids], dim=-1)
        ln = torch.logsumexp(logp[..., self.no_ids], dim=-1)
        return torch.sigmoid(ly - ln), (ly.exp() + ln.exp())

    @torch.no_grad()
    def continue_batch(self, prefix: Prefix, seqs: list[list[int]]):
        """Run B independent continuations of `prefix`. Returns (last-token logits [B, V], cache)."""
        B, Ls = len(seqs), max(len(s) for s in seqs)
        ids = torch.full((B, Ls), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, Ls), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        cache = copy.deepcopy(prefix.cache)
        if B > 1:
            cache.batch_repeat_interleave(B)
        attn = torch.cat([torch.ones((B, prefix.length), dtype=torch.long), mask], dim=1)
        pos = (prefix.next_pos + torch.arange(Ls)).view(1, 1, Ls).expand(3, B, Ls)
        # B == 1 (batched modes, long prompts): keep only the last position's logits
        out = self.model(input_ids=ids.to(self.device), attention_mask=attn.to(self.device),
                         position_ids=pos.to(self.device), past_key_values=cache, use_cache=True,
                         logits_to_keep=1 if B == 1 else 0)
        if B == 1:
            return out.logits[:, -1], out.past_key_values
        lens = mask.sum(1) - 1
        return out.logits[torch.arange(B), lens.to(out.logits.device)], out.past_key_values

    @torch.no_grad()
    def answer_stateless(self, prefix: Prefix, post: str, questions: list[str]):
        """One sequence per question. Returns (p_yes, mass, n_tokens, wall_ms for the whole call)."""
        seqs = [self.encode(q + post) for q in questions]
        _sync()
        t = time.perf_counter()
        logits, _ = self.continue_batch(prefix, seqs)
        p, mass = self.p_yes(logits)
        _sync()
        return p.tolist(), mass.tolist(), [len(s) for s in seqs], (time.perf_counter() - t) * 1e3

    @torch.no_grad()
    def answer_batched(self, prefix: Prefix, rest: str, k: int):
        """k questions in one prompt; the JSON skeleton {"1": "<ans>", "2": ...} is forced and P(yes)
        is read at each answer slot. Returns (p_yes, mass, n_in, n_out, prefill_ms, decode_ms)."""
        _sync()
        t = time.perf_counter()
        first = self.encode(rest + '{"1": "')
        logits, cache = self.continue_batch(prefix, [first])
        _sync()
        prefill_ms = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        pos, ps, masses, n_out = prefix.next_pos + len(first), [], [], 0
        cur_len = prefix.length + len(first)
        for i in range(1, k + 1):
            p, m = self.p_yes(logits[0])
            ps.append(float(p))
            masses.append(float(m))
            if i == k:
                break
            chunk = self.encode(f'{"Yes" if p >= 0.5 else "No"}", "{i + 1}": "')
            n_out += len(chunk)
            ids = torch.tensor([chunk], device=self.device)
            cur_len += len(chunk)
            position = (pos + torch.arange(len(chunk), device=self.device)).view(1, 1, -1).expand(3, 1, -1)
            out = self.model(input_ids=ids, attention_mask=torch.ones((1, cur_len), dtype=torch.long, device=self.device),
                             position_ids=position, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache, logits, pos = out.past_key_values, out.logits[:, -1], pos + len(chunk)
        _sync()
        return ps, masses, len(first), n_out + 1, prefill_ms, (time.perf_counter() - t) * 1e3

    @torch.no_grad()
    def reference_p_yes(self, image: Image.Image, text: str) -> float:
        """Uncached single forward over the full prompt; used to verify the prefix-cache path."""
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1]
        return float(self.p_yes(logits)[0])
