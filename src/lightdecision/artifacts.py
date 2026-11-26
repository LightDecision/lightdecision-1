"""Versioned model artifacts, with explicit completeness and file hashes."""

import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())




def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def load_manifest(path, *, verify_hashes=False):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("artifact_format_version") != 1:
        raise ValueError("unsupported artifact format")
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"incomplete artifact: {path}")
    if (path / "COMPLETE").read_text().strip() != sha256_file(path / "manifest.json"):
        raise ValueError("manifest completeness hash mismatch")
    for name, info in manifest["files"].items():
        entry = path / name
        if not entry.is_relative_to(path) or ".." in Path(name).parts:
            raise ValueError(f"invalid artifact path: {name}")
        if not entry.is_file() or entry.stat().st_size != info["bytes"]:
            raise ValueError(f"artifact file missing or wrong size: {name}")
        if verify_hashes and sha256_file(entry) != info["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
    return manifest




def load_calibration(path, manifest):
    """Read small calibration metadata and verify its weight/data bindings."""
    path = Path(path)
    entry = manifest["files"].get("calibration.json")
    if entry is None or sha256_file(path / "calibration.json") != entry["sha256"]:
        raise ValueError("calibration metadata hash mismatch")
    calibration = read_json(path / "calibration.json")
    temperature = calibration.get("temperature")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    if manifest["stage"] == "calibrated-serving":
        model = calibration.get("model", {})
        data = calibration.get("data", {})
        fit = calibration.get("fit", {})
        provenance = manifest.get("provenance", {})
        weights = {name: info for name, info in manifest["files"].items() if name.endswith(".safetensors")}
        if not weights or model.get("weight_files") != weights:
            raise ValueError("calibration/model weight binding mismatch")
        if calibration.get("fitted") is not True or fit.get("temperature") != temperature:
            raise ValueError("serving artifact requires a fitted, consistent temperature")
        if (not model.get("artifact_manifest_sha256")
                or model["artifact_manifest_sha256"] != provenance.get("base_artifact_manifest_sha256")
                or not data.get("manifest_sha256")
                or data["manifest_sha256"] != provenance.get("calibration_data_manifest_sha256")
                or data.get("split") != "calibration"):
            raise ValueError("calibration provenance binding mismatch")
        from lightdecision.compute import LEGACY_COMPUTE, validate_compute_contract
        declared = validate_compute_contract(read_json(find_config_path(path, manifest)).get("compute_contract", LEGACY_COMPUTE))
        if model.get("compute_contract", LEGACY_COMPUTE) != declared:
            raise ValueError("calibration/model compute contract mismatch")
        if declared != LEGACY_COMPUTE and calibration.get("execution", {}).get("compute_contract") != declared:
            raise ValueError("calibration execution compute contract mismatch")
        if (model.get("backend_id") != manifest["backend_id"]
                or model.get("adapter_api_version") != manifest["adapter_api_version"]
                or model.get("model_config_sha256") != sha256_file(find_config_path(path, manifest))):
            raise ValueError("calibration/model configuration mismatch")
    return calibration


def load_decision_artifact(path, *, device="cpu", dtype="bfloat16", verify_hashes=False):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from lightdecision.config import LightDecisionConfig
    from lightdecision.models.registry import get_backend
    from lightdecision.models.decision.head import DecisionHead
    from lightdecision.models.decision.model import LightDecisionModel
    path = Path(path)
    manifest = load_manifest(path, verify_hashes=verify_hashes)
    if "decision_head.safetensors" not in manifest["files"]:
        raise ValueError("checkpoint has no decision head")
    calibration = load_calibration(path, manifest)
    config = LightDecisionConfig.from_dict(read_json(find_config_path(path, manifest)))
    if (config.backend_id, config.adapter_api_version) != (manifest["backend_id"], manifest["adapter_api_version"]):
        raise ValueError("config/manifest backend mismatch")
    if config.kernel_backend == "fla":
        from lightdecision.models.qwen35.kernels import configure_upstream_fla
        configure_upstream_fla()
    adapter = get_backend(config.backend_id, config.adapter_api_version).build_adapter(path, device=device, dtype=dtype)
    if adapter.get_hidden_size() != config.hidden_size:
        raise ValueError("head/backbone hidden size mismatch")
    head = DecisionHead(config.hidden_size, **config.head).to(device=device)
    head.load_state_dict(load_file(str(path / "decision_head.safetensors")), strict=True)
    model = LightDecisionModel(adapter, head)
    model.lightdecision_config = config
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    return model, tokenizer, config, calibration


def find_config_path(path, manifest):
    """Resolve the checkpoint's model configuration from its manifest.

    The configuration is identified by its schema, so loading does not depend
    on an exporter-specific filename. Exactly one configuration is required.
    """
    path = Path(path)
    required = {"backend_id", "adapter_api_version", "hidden_size", "head", "serializer"}
    matches = []
    for name, info in manifest["files"].items():
        if len(Path(name).parts) != 1 or not name.endswith(".json"):
            continue
        value = read_json(path / name)
        if isinstance(value, dict) and required <= value.keys():
            if sha256_file(path / name) != info["sha256"]:
                raise ValueError("model configuration hash mismatch")
            matches.append(path / name)
    if len(matches) != 1:
        raise ValueError("checkpoint must declare exactly one model configuration")
    return matches[0]
