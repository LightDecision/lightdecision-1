import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lightdecision.artifacts import find_config_path, load_calibration, load_manifest
from lightdecision.runtime.hub import resolve_artifact


def write_checkpoint(root, config_name="lightdecision_config.json"):
    config = {"backend_id": "qwen35_vl", "adapter_api_version": 1,
              "hidden_size": 8, "head": {}, "serializer": {"version": 1}}
    (root / config_name).write_text(json.dumps(config))
    (root / "calibration.json").write_text(json.dumps({"temperature": 1.0, "fitted": False}))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in root.iterdir() if p.is_file()}
    manifest = {"artifact_format_version": 1, "stage": "decision-rlcd", "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "COMPLETE").write_text(hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() + "\n")
    return manifest


def test_config_is_discovered_by_schema_and_validated(tmp_path):
    write_checkpoint(tmp_path, "exporter_model_settings.json")
    manifest = load_manifest(tmp_path, verify_hashes=True)
    assert find_config_path(tmp_path, manifest).name == "exporter_model_settings.json"
    assert load_calibration(tmp_path, manifest)["temperature"] == 1
    config = tmp_path / "exporter_model_settings.json"
    config.write_text(config.read_text().replace('"hidden_size": 8', '"hidden_size": 9'))
    with pytest.raises(ValueError, match="configuration hash"):
        find_config_path(tmp_path, manifest)


def test_incomplete_and_modified_checkpoint_are_rejected(tmp_path):
    write_checkpoint(tmp_path)
    (tmp_path / "COMPLETE").write_text("0" * 64)
    with pytest.raises(ValueError, match="completeness"):
        load_manifest(tmp_path)


def test_local_model_needs_no_hub_access(tmp_path):
    with patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
        assert resolve_artifact(tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError):
        resolve_artifact(tmp_path / "missing")


def test_hub_files_are_pinned_to_one_commit(tmp_path):
    write_checkpoint(tmp_path)
    commit = "a" * 40
    with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download", return_value=str(tmp_path / "manifest.json")) as download, patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snapshot:
        api.return_value.model_info.return_value.sha = commit
        assert resolve_artifact("owner/model", revision="release") == tmp_path
        assert download.call_args.kwargs["revision"] == commit
        assert snapshot.call_args.kwargs["revision"] == commit
        assert set(snapshot.call_args.kwargs["allow_patterns"]) == {
            "manifest.json", "COMPLETE", "lightdecision_config.json", "calibration.json"}


def test_hub_manifest_cannot_request_outside_files(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"../secret": {}}}))
    with patch("huggingface_hub.hf_hub_download", return_value=str(tmp_path / "manifest.json")), patch("huggingface_hub.snapshot_download") as snapshot:
        with pytest.raises(ValueError, match="invalid checkpoint"):
            resolve_artifact("owner/model", local_files_only=True)
        snapshot.assert_not_called()
