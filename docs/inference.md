# Inference guide

## Checkpoints and execution

The Python and HTTP interfaces load the complete checkpoint: vision-language backbone, decision head, tokenizer, processor, configuration, and temperature metadata. Configuration is resolved from the checkpoint manifest. Load the entire directory rather than its `backbone/` subdirectory.

```bash
lightdecision validate-artifact shenjunhao/lightdecision-1-4b
lightdecision validate-artifact /path/to/checkpoint --verify-hashes
```

Full hash verification reads all weight files. `--revision`, `--cache-dir`, and `--local-files-only` are available for loading, serving, and checkpoint validation. An online Hub load resolves one immutable commit before downloading files.

The default device is `cuda:0`. Select another GPU with `--device cuda:1` or `CUDA_VISIBLE_DEVICES`. Use one service process per model instance. The server serializes inference requests through that instance; increasing web-worker count duplicates model memory.

`auto` selects the local runtime's shared-prefix or expanded path according to the backend and request. Shared-prefix execution reuses state/question computation and forks request-local caches for candidate continuations. Images referenced only in the prefix share their visual encoding; images in candidate descriptions use complete visual branches with one branch resident at a time.

The released checkpoints require `bfloat16` backbone loading, FP32 head computation, and `fla-core==0.5.2`. A missing or incompatible kernel installation raises a configuration error. These kernels target CUDA execution.

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

## Images

For a shared image prefix, supply an `images` array containing local file paths or image data URLs. Local paths refer to the machine running inference. For HTTP clients, prefer embedded PNG/JPEG bytes through named assets:

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

## Responses and HTTP configuration

`POST /v1/systemone` returns `model`, `answers`, and `usage`. Answers are keyed by question ID. `usage.output_tokens` is zero because the runtime computes candidate probabilities directly. The Python interface includes execution metadata; HTTP includes it only with `--include-runtime-metrics`.

Set `LIGHTDECISION_API_KEY` to require bearer authentication for `/v1/models` and `/v1/systemone`. If it is unset, the service runs without authentication. `/health` remains a readiness endpoint. The default bind address is `127.0.0.1`; choose `--host 0.0.0.0` when deploying behind your network gateway.

Invalid model IDs, malformed questions, unused assets, or exceeded token budgets return validation errors. To reduce activation memory, lower `--branch-microbatch-size`; this preserves the full candidate set. Image resolution and the complete option-set interaction also contribute to memory use.
