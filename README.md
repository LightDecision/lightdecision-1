# LightDecision-1

LightDecision-1 predicts probabilities over candidate choices supplied with each request. It supports text and image inputs, action selection, binary judgments, and rubric scoring without generating an answer token by token.

This repository provides the inference runtime, Python interface, and HTTP service for both released models.

| Model | Weights |
| --- | --- |
| LightDecision-1-4B | [Hugging Face](https://huggingface.co/shenjunhao/lightdecision-1-4b) |
| LightDecision-1-27B | [Hugging Face](https://huggingface.co/shenjunhao/lightdecision-1-27b) |

## Installation

Use Python 3.11 or later and a CUDA-enabled PyTorch installation with BF16 support. Install the appropriate PyTorch build for your CUDA environment first, then:

```bash
git clone https://github.com/LightDecision/lightdecision-1.git
cd lightdecision-1
python -m pip install -e ".[cuda,serve]"
lightdecision doctor
```

The released checkpoints use BF16 backbone computation and an FP32 decision head. The runtime pins Transformers and Flash Linear Attention to the versions required by its cache and kernel interfaces. Weight files occupy approximately 9.2 GB for 4B and 52.3 GB for 27B; GPU memory must also accommodate image processing, activations, and caches.

Both checkpoints have been checked with Python 3.12, PyTorch 2.11.0 (CUDA 12.9), Transformers 5.12.1, and `fla-core==0.5.2` on an NVIDIA H200, including text decisions, shared image inputs, candidate images, and HTTP requests.

## Run a decision

```bash
lightdecision infer \
  --model shenjunhao/lightdecision-1-4b \
  --request examples/decision.json
```

Use `shenjunhao/lightdecision-1-27b` to select the larger model. Hub checkpoints are downloaded automatically. `--model /path/to/checkpoint` loads local files, and `--revision <commit>` pins a Hub version.

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

## Image inputs

```bash
lightdecision infer \
  --model shenjunhao/lightdecision-1-4b \
  --request examples/vision.json \
  --image /path/to/image.jpg
```

Repeat `--image` for multiple images. The JSON API also supports named image assets inside the state, question, or individual candidate descriptions. See the [inference guide](docs/inference.md) for image assets and all three output types.

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

`GET /health` reports readiness; `GET /v1/models` lists the model identifier and service alias. Each process loads one model on the selected GPU. Use separate ports and `--device` values for multiple model instances.

## Runtime controls

- `--branch-microbatch-size`: bounds the number of candidate branches executed together; default 8.
- `--max-request-tokens`: bounds logical request tokens; default 65,536, also subject to the checkpoint's limits.
- `--execution-mode`: `auto`, `shared_prefix`, or `expanded`.
- `--include-runtime-metrics`: adds timing and execution counters to HTTP responses.

Candidates are scored as a complete set. Their count is determined by the request and available memory; token limits are checked explicitly and inputs are never silently truncated. Reducing the branch microbatch size lowers branch activation memory while retaining every candidate.

See [deployment and request formats](docs/inference.md) for details.

## Inference tests

```bash
python -m pip install -e ".[serve,test]"
pytest -q
```

The tests exercise request parsing, image assets, probability outputs, checkpoint loading, prefix-cache isolation, and HTTP serving without downloading model weights.

## License

[MIT](LICENSE). Model weights and runtime dependencies retain their respective licenses.
