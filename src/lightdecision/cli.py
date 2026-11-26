"""Inference, serving, and checkpoint inspection commands."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        from .runtime.server import main as serve
        return serve(argv[1:])
    parser = argparse.ArgumentParser(prog="lightdecision")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="start the HTTP service; use serve --help")
    commands.add_parser("doctor", help="inspect the inference environment")
    infer = commands.add_parser("infer", help="run one JSON decision request")
    infer.add_argument("--model", default="shenjunhao/lightdecision-1-4b")
    infer.add_argument("--request", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--alias", default="lightdecision-1")
    infer.add_argument("--revision")
    infer.add_argument("--cache-dir")
    infer.add_argument("--local-files-only", action="store_true")
    infer.add_argument("--image", action="append", default=[], help="local image path; repeat for multiple images")
    infer.add_argument("--branch-microbatch-size", type=int, default=8)
    infer.add_argument("--max-request-tokens", type=int, default=65536)
    infer.add_argument("--execution-mode", choices=("auto", "shared_prefix", "expanded"), default="auto")
    validate = commands.add_parser("validate-artifact", help="check a local or Hub checkpoint")
    validate.add_argument("model")
    validate.add_argument("--revision")
    validate.add_argument("--cache-dir")
    validate.add_argument("--local-files-only", action="store_true")
    validate.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        versions = {}
        for name in ("torch", "transformers", "huggingface-hub", "safetensors", "pillow", "fla-core", "fastapi", "uvicorn"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        import torch
        print(json.dumps({"python": sys.version.split()[0], "packages": versions,
                          "cuda_available": torch.cuda.is_available(),
                          "cuda_version": torch.version.cuda}, indent=2))
    elif args.command == "validate-artifact":
        from .artifacts import load_manifest, load_calibration, find_config_path
        from .runtime.hub import resolve_artifact
        root = resolve_artifact(args.model, revision=args.revision, cache_dir=args.cache_dir,
                                local_files_only=args.local_files_only)
        manifest = load_manifest(root, verify_hashes=args.verify_hashes)
        load_calibration(root, manifest)
        find_config_path(root, manifest)
        print(json.dumps({"valid": True, "files": len(manifest["files"]),
                          "hashes_verified": args.verify_hashes}, indent=2))
    elif args.command == "infer":
        from .runtime.engine import DecisionEngine
        body = json.loads(args.request.read_text())
        if not isinstance(body, dict):
            parser.error("request must contain a JSON object")
        body["model"] = args.alias
        if args.image:
            body["images"] = list(body.get("images", [])) + args.image
        engine = DecisionEngine(args.model, device=args.device, alias=args.alias,
                                revision=args.revision, cache_dir=args.cache_dir,
                                local_files_only=args.local_files_only,
                                branch_microbatch_size=args.branch_microbatch_size,
                                max_request_tokens=args.max_request_tokens,
                                execution_mode=args.execution_mode)
        print(json.dumps(engine.predict(body), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
