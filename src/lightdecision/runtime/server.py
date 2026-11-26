"""Serve a frozen decision artifact using the installed FastAPI/uvicorn."""

import argparse
from contextlib import asynccontextmanager
import hmac
import os
import json




def create_app(engine_factory, *, api_key=None, include_runtime_metrics=False):
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from lightdecision.api.typesafe import InvalidDecisionRequest

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine_factory()
        yield
        del app.state.engine

    app = FastAPI(title="LightDecision", lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)):
        if api_key is None:
            return  # Explicit local, unauthenticated deployment.
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health():
        engine = getattr(app.state, "engine", None)
        if engine is None:
            raise HTTPException(503, "model is loading")
        return {"status": "ready", "model": engine.model_id}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        engine = app.state.engine
        return {"object": "list", "data": [{"id": engine.model_id, "object": "model", "owned_by": "lightdecision"},
                                            {"id": engine.alias, "object": "model", "root": engine.model_id, "owned_by": "lightdecision"}]}

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    async def systemone(request: Request):
        from starlette.concurrency import run_in_threadpool
        from lightdecision.api.assets import multipart_request
        content_type = request.headers.get("content-type", "application/json")
        try:
            raw = await request.body()
            body = (multipart_request(raw, content_type) if content_type.lower().startswith("multipart/form-data")
                    else json.loads(raw))
            if not isinstance(body, dict):
                raise ValueError("request must be a JSON object")
        except (ValueError, TypeError) as exc:
            raise HTTPException(400 if content_type.lower().startswith("multipart/form-data") else 422, str(exc)) from exc
        try:
            result = await run_in_threadpool(app.state.engine.predict, body)
            if include_runtime_metrics:
                return result
            return {"model": result["model"], "answers": result["answers"],
                    "usage": {key: result["usage"][key] for key in ("input_tokens", "output_tokens")}}
        except InvalidDecisionRequest as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repository or local checkpoint directory")
    parser.add_argument("--revision", help="Hub commit or tag")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--alias", default="lightdecision-1")
    parser.add_argument("--branch-microbatch-size", type=int, default=8)
    parser.add_argument("--max-request-tokens", type=int, default=65536,
                        help="logical token budget; options/levels have no fixed count limit")
    parser.add_argument("--include-runtime-metrics", action="store_true",
                        help="include LightDecision timing and execution counters in HTTP responses")
    parser.add_argument("--execution-mode", choices=("auto", "shared_prefix", "expanded"), default="auto",
                        help="auto uses shared prefixes on supported backends; expanded is the reference path")
    args = parser.parse_args(argv)
    import uvicorn
    from .engine import DecisionEngine

    app = create_app(lambda: DecisionEngine(args.model, revision=args.revision, cache_dir=args.cache_dir,
                                            local_files_only=args.local_files_only, device=args.device, dtype=args.dtype,
                                            alias=args.alias, branch_microbatch_size=args.branch_microbatch_size,
                                            max_request_tokens=args.max_request_tokens,
                                            execution_mode=args.execution_mode),
                     api_key=os.environ.get("LIGHTDECISION_API_KEY") or None,
                     include_runtime_metrics=args.include_runtime_metrics)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
