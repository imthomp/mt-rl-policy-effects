"""MetricX-24 hybrid in reference-free (QE) mode.

Follows the official google-research/metricx inference: input
"source: {src} candidate: {hyp}", EOS removed from each example *before*
padding, single decoder step with decoder_input_ids = 0, score = logit of
<extra_id_10> (vocab id 250089) clamped to [0, 25], lower is better.

``reward`` returns 1 - raw/25 so that higher is better and the range matches
the COMET-family rewards ([0, 1]).
"""

from __future__ import annotations

import torch

MODEL = "google/metricx-24-hybrid-large-v2p6-bfloat16"
TOKENIZER = "google/mt5-large"  # the bf16 checkpoint ships without a tokenizer
EXTRA_ID_10 = 250089
MAX_LENGTH = 1536


class MetricXQE:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer, MT5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def raw(self, sources: list[str], hypotheses: list[str], batch_size: int = 32) -> list[float]:
        texts = [f"source: {s} candidate: {h}" for s, h in zip(sources, hypotheses)]
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            ids = [self.tokenizer(t, max_length=MAX_LENGTH, truncation=True)["input_ids"]
                   for t in texts[i:i + batch_size]]
            # Remove EOS per example, then pad (official order).
            ids = [x[:-1] if x and x[-1] == self.tokenizer.eos_token_id else x for x in ids]
            width = max(len(x) for x in ids)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=self.device)
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=self.device)
            decoder = torch.zeros(len(ids), 1, dtype=torch.long, device=self.device)
            logits = self.model(input_ids=input_ids, attention_mask=mask,
                                decoder_input_ids=decoder).logits
            out += logits[:, 0, EXTRA_ID_10].float().clamp(0, 25).cpu().tolist()
        return out

    def reward(self, sources: list[str], hypotheses: list[str], batch_size: int = 32) -> list[float]:
        return [1.0 - r / 25.0 for r in self.raw(sources, hypotheses, batch_size)]
