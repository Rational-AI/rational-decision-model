"""Candidate-logit readout: turn the next-token logits into a distribution over the decision options.

Each option is presented to the model under a single-character label (A, B, C, ...). A label can be spelled by
more than one vocabulary entry: Gemma 4 has both a plain "A" token and a byte-fallback token that decodes to "A".
The readout therefore scores an option by log-sum-exp over every token id that decodes exactly to its label, then
normalises over the options only — the same result as masking the vocabulary to the labels and renormalising.
"""
from __future__ import annotations

import string

import torch

LABELS = string.ascii_uppercase


def label_token_ids(tokenizer, n: int = len(LABELS)) -> list[list[int]]:
    """For each of the first n labels, every token id that decodes to exactly that label."""
    wanted = {label: i for i, label in enumerate(LABELS[:n])}
    ids: list[list[int]] = [[] for _ in range(n)]
    for token_id in range(len(tokenizer)):
        i = wanted.get(tokenizer.decode([token_id]))
        if i is not None:
            ids[i].append(token_id)
    missing = [LABELS[i] for i, t in enumerate(ids) if not t]
    if missing:
        raise ValueError(f"the tokenizer has no token for labels {missing}")
    return ids


def option_scores(logits: torch.Tensor, ids: list[list[int]]) -> torch.Tensor:
    """s_i = log sum_{t in T_i} exp(logit_t), one score per option; every option's mass is observed."""
    return torch.stack([torch.logsumexp(logits[t], dim=0) for t in ids]).cpu()


def distribution(scores: torch.Tensor, temperature: float = 1.0) -> list[float]:
    """p_i ∝ exp(s_i / T), which equals the raw p_i^(1/T) renormalised; T = 1 is the raw distribution."""
    return torch.softmax(scores.double() / temperature, dim=0).tolist()
