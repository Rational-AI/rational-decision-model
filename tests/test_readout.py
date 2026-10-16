import math

import torch

from rmdm.readout import distribution, label_token_ids, option_scores


class FakeTokenizer:
    """Ids 0..5 decode to: A, B, <0x41> (a second "A"), " A", "C", "Ab"."""
    vocab = ["A", "B", "A", " A", "C", "Ab"]

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        return self.vocab[ids[0]]


def test_duplicate_label_tokens_are_summed():
    ids = label_token_ids(FakeTokenizer(), 3)
    assert ids == [[0, 2], [1], [4]]  # " A" and "Ab" are not the label A
    logits = torch.tensor([0.0, 0.0, 0.0, 9.0, 0.0, 9.0])
    p = distribution(option_scores(logits, ids))
    # A has two tokens with equal logits, so twice B's (and C's) mass; the excluded tokens carry none
    assert math.isclose(p[0], 0.5) and math.isclose(p[1], 0.25) and math.isclose(p[2], 0.25)


def test_readout_equals_masked_renormalised_softmax():
    ids = [[0, 2], [1], [4]]
    logits = torch.randn(6)
    full = torch.softmax(logits.double(), 0)
    masked = [full[t].sum().item() for t in ids]
    masked = [m / sum(masked) for m in masked]
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(distribution(option_scores(logits, ids)), masked))  # float32 logits


def test_readout_is_order_equivariant():
    ids = [[0, 2], [1], [4]]
    logits = torch.randn(6)
    p = distribution(option_scores(logits, ids))
    q = distribution(option_scores(logits, [ids[2], ids[0], ids[1]]))
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip([p[2], p[0], p[1]], q))


def test_temperature_is_power_then_renormalise():
    scores = torch.tensor([2.0, 0.5, -1.0])
    raw = distribution(scores)
    t = 3.4
    z = [x ** (1 / t) for x in raw]
    assert all(math.isclose(a, b / sum(z), rel_tol=1e-9) for a, b in zip(distribution(scores, t), z))
    assert distribution(scores, 1.0) == raw
