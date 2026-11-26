"""Resolve a local checkpoint or one immutable Hub snapshot."""
import json
from pathlib import Path, PurePosixPath
import re


def resolve_artifact(model, *, revision=None, cache_dir=None, local_files_only=False):
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or str(model).startswith(("./", "../", "~")):
        raise FileNotFoundError(f"checkpoint directory does not exist: {model}")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", str(model)):
        raise ValueError("model must be a local directory or a Hub owner/repository")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    resolved = revision
    if not local_files_only:
        resolved = HfApi().model_info(str(model), revision=revision).sha
    manifest_path = Path(hf_hub_download(
        str(model), "manifest.json", revision=resolved, cache_dir=cache_dir,
        local_files_only=local_files_only,
    ))
    manifest = json.loads(manifest_path.read_text())
    names = list(manifest["files"])
    for name in names:
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError(f"invalid checkpoint file path: {name}")
    return Path(snapshot_download(
        str(model), revision=resolved, cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=["manifest.json", "COMPLETE", *names],
    ))
