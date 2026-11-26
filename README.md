<p align="center">
  <img src="assets/lightdecision-1.svg" width="540" alt="LightDecision-1" />
</p>

<p align="center"><strong>Text &amp; Vision Decision Models</strong><br />Choose · Judge · Score</p>

<p align="center">
  <a href="https://huggingface.co/shenjunhao/lightdecision-1-4b"><img src="assets/model-4b.svg" alt="Hugging Face · 4B Weights" /></a>
  <a href="https://huggingface.co/shenjunhao/lightdecision-1-27b"><img src="assets/model-27b.svg" alt="Hugging Face · 27B Weights" /></a>
  <a href="#run-a-decision"><img src="assets/quick-start.svg" alt="Quick Start" /></a>
  <a href="#http-service"><img src="assets/http-api.svg" alt="HTTP API" /></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#request-format">Request format</a> ·
  <a href="#image-inputs">Image inputs</a> ·
  <a href="#runtime-controls">Runtime controls</a>
</p>

---

**LightDecision-1** predicts probabilities over candidate choices supplied with each request. It supports text and image inputs, action selection, binary judgments, and rubric scoring without generating an answer token by token. This repository provides the inference runtime, Python interface, and HTTP service for both released models.

## Installation

Use Python 3.11 or later and a CUDA-enabled PyTorch installation with BF16 support. Install the appropriate PyTorch build for your CUDA environment first, then:

```bash
git clone https://github.com/LightDecision/lightdecision-1.git
cd lightdecision-1
python -m pip install -e ".[cuda,serve]"
lightdecision doctor
```

The released checkpoints require `bfloat16` backbone loading, FP32 head computation, and `fla-core==0.5.2`. The runtime pins Transformers and Flash Linear Attention to the versions required by its cache and CUDA kernel interfaces; a missing or incompatible kernel installation raises a configuration error. Weight files occupy approximately 9.2 GB for 4B and 52.3 GB for 27B; GPU memory must also accommodate image processing, activations, and caches.

Both checkpoints have been checked with Python 3.12, PyTorch 2.11.0 (CUDA 12.9), Transformers 5.12.1, and `fla-core==0.5.2`, including text decisions, shared image inputs, candidate images, and HTTP requests.

## Run a decision

```bash
lightdecision infer \
  --model shenjunhao/lightdecision-1-4b \
  --request examples/decision.json
```

Use `shenjunhao/lightdecision-1-27b` to select the larger model. Hub checkpoints are downloaded automatically.

```python
from lightdecision.runtime.engine import DecisionEngine

engine = DecisionEngine("shenjunhao/lightdecision-1-4b")
result = engine.predict({
    "model": "lightdecision-1",
    "state": "A robot needs to fill an empty mug at a sink.",
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Choose the next action.",
            "criteria": {
                "Move the mug under the tap": None,
                "Open a drawer": None,
            },
        }
    },
})
print(result["answers"]["action"]["choice"])
print(result["answers"]["action"]["probabilities"])
```

## Checkpoint loading

`--model /path/to/checkpoint` loads local files. The Python and HTTP interfaces load the complete checkpoint: vision-language backbone, decision head, tokenizer, processor, configuration, and temperature metadata. Configuration is resolved from the checkpoint manifest. Load the entire directory rather than its `backbone/` subdirectory.

```bash
lightdecision validate-artifact shenjunhao/lightdecision-1-4b
lightdecision validate-artifact /path/to/checkpoint --verify-hashes
```

Full hash verification reads all weight files. `--revision <commit>`, `--cache-dir`, and `--local-files-only` are available for loading, serving, and checkpoint validation. An online Hub load resolves one immutable commit before downloading files.

The default device is `cuda:0`. Select another GPU with `--device cuda:1` or `CUDA_VISIBLE_DEVICES`.

## Request format

```json
{
  "model": "lightdecision-1",
  "state": "The situation to assess.",
  "questions": {
    "task_id": {
      "type": "choice",
      "instructions": "Choose the best candidate.",
      "criteria": {"Candidate A": "Description A", "Candidate B": null}
    }
  }
}
```

`state`, `instructions`, and candidate descriptions accept strings or structured JSON. Question IDs identify outputs and do not enter the model's decision text. Multiple questions may share one state. The `model` value must match the alias (`lightdecision-1` by default) or the identifier returned by `/v1/models`.

| Type | Criteria | Output |
| --- | --- | --- |
| `choice` | Map of candidate names to descriptions or `null` | Selected `choice`, per-candidate `probabilities`, and `confidence` |
| `noul` | Optional descriptions under `false` and `true` | `noul`: probability of true |
| `score` | Ordered array of rubric-level descriptions | `score`: expected zero-based level index, `probabilities`, `legend`, and `confidence` |

For example, three score levels produce a score between 0 and 2. `confidence` summarizes the returned distribution; it is not a separate prediction.

## Image inputs

```bash
lightdecision infer \
  --model shenjunhao/lightdecision-1-4b \
  --request examples/vision.json \
  --image /path/to/image.jpg
```

Repeat `--image` for multiple images. For a shared image prefix in a JSON request, supply an `images` array containing local file paths or image data URLs. Local paths refer to the machine running inference. For HTTP clients, prefer embedded PNG/JPEG bytes through named assets:

```python
import base64
from pathlib import Path

request = {
    "model": "lightdecision-1",
    "assets": {
        "@scene": {
            "type": "image",
            "format": "jpg",
            "data_base64": base64.b64encode(Path("scene.jpg").read_bytes()).decode(),
        }
    },
    "state": {"observation": "@scene", "goal": "Find a clear walking path."},
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Choose a direction.",
            "criteria": {"Go left": None, "Go right": None, "Stay still": None},
        }
    },
}
```

References may occur in the state, instructions, or candidate descriptions. Repeated references retain their positions. Every declared asset must be used. `data_base64` contains standard padded base64 without a data-URL prefix. Formats are `png`, `jpg`, or `jpeg`.

Multipart HTTP requests put the JSON `request` part first. An asset can use `"part": "image_file"` in place of `data_base64`, followed by a file part named `image_file`. Each file part must be referenced exactly once by an asset.

## HTTP service

```bash
export LIGHTDECISION_API_KEY='your-service-key'
lightdecision serve \
  --model shenjunhao/lightdecision-1-4b \
  --host 127.0.0.1 --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $LIGHTDECISION_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/decision.json
```

`GET /health` reports readiness; `GET /v1/models` lists the model identifier and service alias. `POST /v1/systemone` returns `model`, `answers`, and `usage`. Answers are keyed by question ID. `usage.output_tokens` is zero because the runtime computes candidate probabilities directly. The Python interface includes execution metadata; HTTP includes it only with `--include-runtime-metrics`.

Set `LIGHTDECISION_API_KEY` to require bearer authentication for `/v1/models` and `/v1/systemone`. If it is unset, the service runs without authentication. `/health` remains a readiness endpoint. The default bind address is `127.0.0.1`; choose `--host 0.0.0.0` when deploying behind your network gateway.

Each process loads one model on the selected GPU and serializes inference requests through that instance. Increasing web-worker count duplicates model memory. Use separate ports and `--device` values for multiple model instances.

Invalid model IDs, malformed questions, unused assets, or exceeded token budgets return validation errors.

## Runtime controls

- `--branch-microbatch-size`: bounds the number of candidate branches executed together; default 8.
- `--max-request-tokens`: bounds logical request tokens; default 65,536, also subject to the checkpoint's limits.
- `--execution-mode`: `auto`, `shared_prefix`, or `expanded`.
- `--include-runtime-metrics`: adds timing and execution counters to HTTP responses.

`auto` selects shared-prefix or expanded execution according to the backend and request. Shared-prefix execution reuses state/question computation and forks request-local caches for candidate continuations. Images referenced only in the prefix share their visual encoding; images in candidate descriptions use complete visual branches with one branch resident at a time.

Candidates are scored as a complete set. Their count is determined by the request and available memory; token limits are checked explicitly and inputs are never silently truncated. Reducing the branch microbatch size lowers branch activation memory while retaining every candidate. Image resolution and the complete option-set interaction also contribute to memory use.

## Inference tests

```bash
python -m pip install -e ".[serve,test]"
pytest -q
```

The tests exercise request parsing, image assets, probability outputs, checkpoint loading, prefix-cache isolation, and HTTP serving without downloading model weights.

## License

[MIT](LICENSE). Model weights and runtime dependencies retain their respective licenses.
