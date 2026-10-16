# Rational Multimodal Decision Model (RMDM)

A closed decision over multimodal context, answered by reading a **frozen** Gemma 4's own next-token logits over the
option letters. There is no generated explanation, no JSON parsing and no fine-tuning: one forward pass and one
output position per decision.

```
images + text + JSON state ─▶ Gemma 4 (frozen) ─▶ logits at the answer position
    ─▶ log-sum-exp over every token spelling each option letter ─▶ softmax over the options
    ─▶ temperature calibration ─▶ typed decision + raw and calibrated probabilities
```

Everything runs locally: local weights, local images, local calibration data, no outbound calls
(`HF_HUB_OFFLINE=1` once the weights are cached).

## Run

```bash
uv sync
uv run hf download mlx-community/gemma-4-12B-it-4bit --revision 73bcf09092aa277861d5a191b989b666f7f32e8f
HF_HUB_OFFLINE=1 uv run python -m rmdm.server       # http://127.0.0.1:8009
uv run pytest                                        # readout math, no model needed
HF_HUB_OFFLINE=1 uv run python scripts/smoke.py      # does the image change the decision?
HF_HUB_OFFLINE=1 uv run python scripts/quick_eval.py ../jevbench   # 2-minute text sanity check
```

| env | default | |
|---|---|---|
| `RMDM_BACKEND` | `mlx` on Apple silicon | `mlx` (quantized MLX weights) or `torch` (bf16 reference, Transformers) |
| `RMDM_MODEL` / `RMDM_REVISION` | 12B-it, pinned (4-bit on mlx, bf16 on torch) | e.g. `mlx-community/gemma-4-e4b-it-8bit` |
| `RMDM_PROFILE` | `gemma4-12b-4bit-mlx` | calibration profile, from `src/rmdm/calibration.json` (`raw` = off) |
| `RMDM_SYSTEMONE_TEMPERATURE` | `1.0` | temperature for the JevBench endpoint |
| `RMDM_PORT` | `8009` | |

## API

`POST /v1/decisions`

```json
{
  "input": [
    {"type": "image", "label": "Image 1: current inspection", "image": "<base64 or data: URI>"},
    {"type": "text", "text": "Inspection performed after 2,300 operating hours."},
    {"type": "json", "value": {"machine": "A17", "temperature": 83.2}}
  ],
  "decision": {
    "id": "maintenance_action",
    "instructions": "Select the safest appropriate action.",
    "options": [
      {"id": "continue", "description": "Continue normal operation."},
      {"id": "inspect", "description": "Schedule a manual inspection."},
      {"id": "stop", "description": "Stop operation immediately."}
    ],
    "abstain": true
  },
  "calibration_profile": "gemma4-12b-4bit-mlx"
}
```

Images can also go anywhere text goes, as in TypeSafe's API: list each image once in a top-level `assets` map
under a reference string, and write the reference in the input, in a question's `instructions` or in an option's
`description` (define an option by an example picture). Several decisions over the same input go in a
`decisions` map, and come back as `answers` keyed the same way; the input is computed once and reused (prefix KV
cache) when the question itself holds no image. Types: `choice` (default), `noul` (`criteria` with `true` /
`false`, answer adds `noul` = P(true)) and `score` (`options` are ordered levels, lowest first; answer adds
`expected_level`) and `box` (`instructions` name the target, `image` names the asset; answer: `found` =
P(visible), `box` = left/top/right/bottom as fractions of the image, `confidence`). A box is built from the same
one-token readout: a grid is drawn round the picture, four lettered-option questions ask for the column or row of
each edge, then the found region is zoomed and asked again (9 readouts, about 7.5 s on the M1 Max). On synthetic
street scenes with known sign positions it reaches a mean IoU of 0.67 (7 of 8 above 0.5) and reports an absent
target as `found` 0.02. See `examples/12-assets-and-several-decisions.json`.

`abstain: true` appends a `human_review` option. The distribution is always conditional on the options offered: if
none fits, the mass still goes somewhere, so any task that can see incomplete or out-of-distribution evidence
should offer a way out.

The response carries `decision`, `probabilities` (calibrated), `raw_probabilities`, `calibration`, `latency_ms`,
`usage` and a `trace` (backend, model and revision, prompt version, option ids, SHA-256 of the input, timestamp)
so every decision can be replayed. Images are accepted only inline: no URLs, no server-side paths.

`examples/` holds eleven ready requests, six with images embedded (drawn locally, no real data) and five text or
JSON only:

```bash
curl -s localhost:8009/v1/decisions -H 'Content-Type: application/json' -d @examples/07-ticket-triage.json | jq '{decision, probabilities}'
```

`POST /v1/systemone` speaks JevBench's `typesafe` wire format (text only) for the benchmark baseline.

## How the readout works

- Options are presented as letters A, B, C, ... (2 to 26 options) with Cygnet's system prompt and layout, word
  for word, so text-only figures are comparable with Cygnet's.
- Gemma 4's vocabulary spells every capital letter with **two** token ids (a plain token and a byte-fallback
  token). An option's score is `log Σ exp(logit_t)` over every id that decodes exactly to its letter, and the
  distribution is the softmax of those scores. That is the same as masking the vocabulary to the letters and
  renormalising, and every option's mass is observed (no `top_logprobs` cut-off).
- Calibration is `p_i^(1/T)` renormalised, per profile. Fit `T` on held-out items only (`scripts/calibrate.py`).
  For 12B 4-bit, `T = 3.7` by NLL on Cygnet's 241 generated items (none from JevBench); fitted on one half, it
  lowers expected calibration error on the other half from 0.184 to 0.108. Cygnet's 12B bf16: `T = 3.4`,
  0.140 to 0.101. A temperature is model-specific: refit it when the model or its quantization changes.
- Thinking is off: a reasoning trace before the answer slot would not be a readout.

## JevBench baseline (text)

```bash
git clone https://github.com/fstandhartinger/jevbench && git -C jevbench checkout 2fa63fa
HF_HUB_OFFLINE=1 uv run python -m rmdm.server &
cd jevbench && python3 -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --key-env '' --model rmdm \
  --cost-basis self_hosted_gpu --reserve-usd 0 --delay-s 0 --run-label rmdm \
  --results OUT/results.jsonl --raw-dir OUT/raw --ledger OUT/ledger.jsonl --manifest OUT/manifest.json
```

`uv run python scripts/summarize.py OUT/results.jsonl` prints accuracy by tier and latency. Cygnet's reference,
203/231, is Gemma 4 12B in bf16 on vLLM on a 48 GB datacentre GPU; the runs here are quantized MLX weights on an
M1 Max (32 GB), frozen, with the same prompt and readout:

| run | correct | easy · standard · hard | standard p50 |
|---|---|---|---|
| Cygnet, 12B bf16, vLLM, L40S | 203/231 (87.9%) | 48/48 · 70/72 · 85/111 | 0.050 s |
| 12B 4-bit, MLX, M1 Max (`runs/12b-4bit-mlx`) | 193/231 (83.5%) | 48/48 · 67/72 · 78/111 | 0.847 s |

The 12B 4-bit run sees the same prompts as Cygnet (704 mean input tokens on both) and gives the same answer on
210 of 231 items; the gap is on the hard tier and is most likely the 4-bit quantization. On the short tiers alone
(`scripts/quick_eval.py`), E4B 8-bit scores 117/120, 12B 4-bit 115/120 and Cygnet 118/120.

Laptop latency is prefill-bound: about 170 prompt tokens/s for 12B 4-bit on an M1 Max, so a 4k-token hard item
takes about 20 s where an L40S takes a fraction of a second.

## Provenance and licences

This repository is licensed under Apache-2.0 (`LICENSE`); `NOTICE` lists what comes from elsewhere. It contains no
model weights: the runtime downloads Gemma 4 from Hugging Face on first use.

| component | licence | notes |
|---|---|---|
| Gemma 4 weights (`google/gemma-4-*` and the `mlx-community` conversions) | Apache-2.0 | Also subject to Google's [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which bars automated decisions in domains that affect material or individual rights. A deployment in such a domain needs a human in the loop, or a different model. |
| readout approach | — | [NInfer](https://github.com/igorls/ninfer) (Apache-2.0) originated the one-pass option-logit readout; [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) (MIT) did it over vLLM. The system prompt and option layout here are Cygnet's, reused verbatim for comparability; the code is our own. |
| JevBench | MIT | used unmodified, as an external tool |
| Transformers · MLX · mlx-vlm · PyTorch | Apache-2.0 · MIT · MIT · BSD-3 | |
