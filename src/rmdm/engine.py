"""Frozen Gemma 4 on the local machine: one forward pass per decision, logits read at the answer position.

Two backends behind one interface. The model is chosen with RMDM_BACKEND / RMDM_MODEL / RMDM_REVISION:
  mlx    (default on Apple silicon) quantized MLX weights, fits next to a running dev stack
  torch  the unquantized bf16 reference weights through Hugging Face Transformers
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .readout import LABELS, label_token_ids, option_scores

DEFAULTS = {
    "mlx": ("mlx-community/gemma-4-12B-it-4bit", "73bcf09092aa277861d5a191b989b666f7f32e8f"),
    "torch": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
}

# The system prompt and the lettered layout are Cygnet's (blockbrain-ai/cygnet-recipe, MIT), kept word for word so
# the text-only readout is comparable with its published JevBench figures. Bump PROMPT_VERSION on any change: it is
# recorded with every decision.
PROMPT_VERSION = "letters-v1"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)


def render_state(state) -> str:
    """Structured state is rendered with indent=1, the rendering Cygnet's figures were measured with."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def decision_text(state_text: str, instructions: str, descriptions: list[str]) -> str:
    lines = [state_text.rstrip(), "", instructions.rstrip(), "", "Options:"]
    lines += [f"{LABELS[i]}. {d}" for i, d in enumerate(descriptions)]
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines)


class Engine:
    backend: str
    # A reasoning trace before the answer slot would not be a readout: thinking stays off.
    TEMPLATE = {"add_generation_prompt": True, "enable_thinking": False}

    def __init__(self, model_id: str, revision: str):
        self.model_id, self.revision = model_id, revision
        self.label_ids = label_token_ids(self.tokenizer)

    def scores(self, text: str, images: list, n_options: int, shared_chars: int = 0) -> tuple[torch.Tensor, int]:
        """Option scores for a user turn whose text ends with n_options lettered options.

        text holds one IMAGE marker (self.image_marker) where each of images goes, in order. It is sent as a
        single text part: chat templates disagree on how to join several parts and where to put images (the
        MLX conversion's template moves every image to the front of the turn), so the placement is ours.
        text[:shared_chars] is what further decisions on the same input repeat (the state); a backend may cache
        it. Returns the per-option log-sum-exp scores and the prompt length in tokens.
        """
        if not 2 <= n_options <= len(LABELS):
            raise ValueError(f"a decision needs 2 to {len(LABELS)} options, got {n_options}")
        if text.count(self.image_marker) != len(images):
            raise ValueError("the number of image markers and images differ")
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, **self.TEMPLATE)
        logits, n_tokens = self.next_token_logits(prompt, images, text, shared_chars)
        return option_scores(logits, self.label_ids[:n_options]), n_tokens


class TorchEngine(Engine):
    backend = "torch"

    def __init__(self, model_id: str, revision: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.tokenizer = self.processor.tokenizer
        self.image_marker = self.processor.image_token
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16).to(device).eval()
        super().__init__(model_id, revision)

    @torch.inference_mode()
    def next_token_logits(self, prompt, images, text, shared_chars):
        inputs = self.processor(text=prompt, images=images or None, add_special_tokens=False,
                                return_tensors="pt").to(self.model.device)
        return self.model(**inputs, logits_to_keep=1).logits[0, -1].float(), inputs["input_ids"].shape[1]


class MlxEngine(Engine):
    backend = "mlx"

    def __init__(self, model_id: str, revision: str):
        import mlx.core as mx
        from mlx_vlm import load
        self.model, self.processor = load(model_id, revision=revision)
        # Keep the weights resident on the GPU for the life of the engine, as mlx-vlm does around a generation.
        # Under memory pressure macOS otherwise pages them out between decisions and every forward pass pays to
        # read them back.
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.image_marker = self.processor.image_token
        # The KV cache of the last shared prefix (system prompt, state and its images): (key, n_tokens, snapshot).
        # Further decisions on the same input then pay only for their own question. One entry, keyed on the exact
        # prefix text and image bytes; an LRU would serve interleaved requests over several inputs.
        self._prefix = None
        super().__init__(model_id, revision)

    def _prepare(self, prompt, images):
        from mlx_vlm.utils import prepare_inputs
        return prepare_inputs(self.processor, images=images or None, prompts=prompt,
                              image_token_index=getattr(self.model.config, "image_token_index", None))

    def _forward(self, inputs, cache=None, start=0):
        """Run the language model over inputs["input_ids"][:, start:], extending cache; last-position logits."""
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
        if start:  # a text-only suffix after a cached prefix that holds every image
            ids, pixels, extra = inputs["input_ids"][:, start:], None, {}
        else:
            ids, pixels = inputs["input_ids"], inputs.get("pixel_values")
        embedded = self.model.get_input_embeddings(ids, pixels, **extra)
        # Called in two steps, as mlx-vlm's own generation does, because the model's __call__ (12B) drops
        # mm_token_type_ids, which the language model needs to let an image's tokens attend to each other.
        # No mask: one unpadded sequence, so the model builds its own causal mask (plus that image overlay); the
        # processor's all-ones attention_mask, passed as `mask`, would replace it and let every position see the
        # future. logits_to_keep=1: only the answer position is projected onto the 262k vocabulary.
        out = self.model.language_model(
            None, inputs_embeds=embedded.inputs_embeds, per_layer_inputs=embedded.per_layer_inputs,
            mm_token_type_ids=extra.get("mm_token_type_ids"), cache=cache, logits_to_keep=1)
        return getattr(out, "logits", out)[0, -1]

    def _cached_prefix(self, prompt, text, shared_chars, images, input_ids):
        """(cache, n_tokens) holding the prompt up to text[:shared_chars], or (None, 0) if it cannot be split.

        Only a prefix that holds every image is cached, and only when it tokenizes to exactly the first tokens of
        the full prompt, so the model sees the same token sequence with or without the cache.
        """
        import mlx.core as mx
        at = prompt.rfind(text)
        if at < 0 or self.image_marker in text[shared_chars:]:
            return None, 0
        prefix = prompt[: at + shared_chars]
        key = (prefix, tuple(hashlib.sha256(i.tobytes()).hexdigest() for i in images))
        if self._prefix is None or self._prefix[0] != key:
            inputs = self._prepare(prefix, images)
            n = inputs["input_ids"].shape[1]
            if n >= input_ids.shape[1] or not mx.array_equal(inputs["input_ids"], input_ids[:, :n]).item():
                return None, 0
            cache = self.model.language_model.make_cache()
            mx.eval(self._forward(inputs, cache))
            snapshot = [(tuple(mx.contiguous(mx.array(a)) for a in c.state), c.meta_state) for c in cache]
            mx.eval([a for s, _ in snapshot for a in s])
            self._prefix = (key, n, snapshot)
        _, n, snapshot = self._prefix
        cache = self.model.language_model.make_cache()
        for c, (state, meta) in zip(cache, snapshot):
            c.state, c.meta_state = state, meta
        return cache, n

    def next_token_logits(self, prompt, images, text, shared_chars):
        import mlx.core as mx
        inputs = self._prepare(prompt, images)
        cache, start = None, 0
        if shared_chars:
            cache, start = self._cached_prefix(prompt, text, shared_chars, images, inputs["input_ids"])
        logits = self._forward(inputs, cache, start).astype(mx.float32)
        return torch.from_numpy(np.array(logits)), inputs["input_ids"].shape[1]


def load_engine() -> Engine:
    backend = os.environ.get("RMDM_BACKEND") or ("mlx" if torch.backends.mps.is_available() else "torch")
    model_id, revision = DEFAULTS[backend]
    model_id = os.environ.get("RMDM_MODEL", model_id)
    revision = os.environ.get("RMDM_REVISION", revision if model_id == DEFAULTS[backend][0] else "main")
    return (MlxEngine if backend == "mlx" else TorchEngine)(model_id, revision)
