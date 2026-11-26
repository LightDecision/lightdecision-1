"""Decision serving with shared-prefix execution and an expanded reference mode."""

from pathlib import Path
from threading import Lock
import time

import torch

from lightdecision.api.typesafe import InvalidDecisionRequest, format_answer, parse_request
from lightdecision.artifacts import load_decision_artifact, sha256_file
from lightdecision.compute import validate_inference_compute
from .collator import collate_serialized
from .hub import resolve_artifact
from lightdecision.models.decision.serializer import DecisionSerializer
from lightdecision.models.registry import get_backend
from .prefix import predict_shared_prefix, predict_one_prefix


class DecisionEngine:
    def __init__(self, artifact, *, device="cuda:0", dtype="bfloat16", alias="lightdecision-1",
                 branch_microbatch_size=8, max_request_tokens=65536, execution_mode="auto",
                 revision=None, cache_dir=None, local_files_only=False):
        if execution_mode not in {"auto", "expanded", "shared_prefix"}:
            raise ValueError("execution_mode must be auto, expanded or shared_prefix")
        if branch_microbatch_size < 1:
            raise ValueError("branch_microbatch_size must be positive")
        if max_request_tokens < 1:
            raise ValueError("max_request_tokens must be positive")
        artifact = resolve_artifact(artifact, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
        self.device = torch.device(device)
        self.model, self.tokenizer, self.config, calibration = load_decision_artifact(artifact, device=device, dtype=dtype)
        self.compute_contract = validate_inference_compute(self.config, dtype)
        capabilities = get_backend(self.config.backend_id, self.config.adapter_api_version).capabilities()
        supports_prefix = (capabilities.causal_prefix and capabilities.prefix_reuse
                           and capabilities.fork and capabilities.multi_token_continuation)
        if execution_mode == "shared_prefix" and not supports_prefix:
            raise ValueError(f"backend {self.config.backend_id!r} does not support shared-prefix inference")
        self.execution_mode = ("shared_prefix" if supports_prefix else "expanded") if execution_mode == "auto" else execution_mode
        self.auto_route = execution_mode == "auto" and supports_prefix and hasattr(self.model.adapter, "text_model")
        self.model.eval()
        self.model.branch_microbatch_size = branch_microbatch_size
        self.temperature = float(calibration["temperature"])
        args = dict(self.config.serializer)
        args.pop("version")
        self.serializer = DecisionSerializer(self.tokenizer, **args)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.model_id = "lightdecision-" + sha256_file(Path(artifact) / "manifest.json")[:16]
        self.alias = alias
        self.max_request_tokens = max_request_tokens
        self.lock = Lock()

    @torch.inference_mode()
    def predict(self, body):
        try:
            questions = parse_request(body, model_ids={self.alias, self.model_id})
            has_assets = any(q.prefix_images or any(q.option_images) for q in questions)
            if has_assets:
                from .multimodal import serialize_asset_question, serialize_images
                prepared = []
                for question in questions:
                    if question.prefix_images or any(question.option_images):
                        prepared.append(serialize_asset_question(self, question, body.get("images", ())))
                    elif body.get("images"):
                        encoded, media = serialize_images(self, question.decision, body["images"])
                        prepared.append((encoded, media, None))
                    else:
                        prepared.append((self.serializer(question.decision), {}, None))
                serialized = [x[0] for x in prepared]
            elif body.get("images"):
                from .multimodal import serialize_images
                prepared = [(*serialize_images(self, question.decision, body["images"]), None) for question in questions]
                serialized = [x[0] for x in prepared]
            else:
                prepared = None
                serialized = [self.serializer(question.decision) for question in questions]
        except (ValueError, TypeError) as exc:
            raise InvalidDecisionRequest(str(exc)) from exc
        # Keep expanded_input_tokens as the unpadded expanded reference budget.
        # executed_input_tokens includes any padding actually sent to the model.
        logical_tokens = len(serialized[0].state_ids) + sum(item.logical_token_count - len(item.state_ids) for item in serialized)
        if prepared is not None:
            logical_tokens = sum(item.logical_token_count for item in serialized)
        if logical_tokens > self.max_request_tokens:
            raise InvalidDecisionRequest(f"request uses {logical_tokens} tokens, exceeds {self.max_request_tokens}; truncation is disabled")
        answers = {}
        expanded_tokens = sum(len(branch.input_ids) for item in serialized for branch in item.branches)
        executed_tokens = 0
        execution = self.execution_mode
        # Measured crossover: cache creation/forks cost more than recomputation on short inputs.
        if self.auto_route and prepared is None and expanded_tokens - logical_tokens < 4096:
            execution = "expanded"
        if prepared is not None:
            execution = "multimodal_branches" if any(x[2] is not None for x in prepared) else "shared_prefix"
        with self.lock:
            start = time.perf_counter()
            if prepared is not None:
                outputs = []
                for encoded, media, branch_media in prepared:
                    if branch_media is not None:
                        from .multimodal import predict_asset_branches
                        value, count = predict_asset_branches(self, encoded, branch_media)
                        outputs.append(value)
                    else:
                        media = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in media.items()}
                        values, count = predict_one_prefix(self.model, encoded, device=self.device,
                            branch_microbatch_size=self.model.branch_microbatch_size, pad_token_id=self.pad_id,
                            temperature=self.temperature, media=media)
                        outputs.extend(values)
                    executed_tokens += count
            elif execution == "shared_prefix":
                outputs, executed_tokens = predict_shared_prefix(
                    self.model, serialized, device=self.device,
                    branch_microbatch_size=self.model.branch_microbatch_size,
                    pad_token_id=self.pad_id,
                    temperature=self.temperature,
                )
            else:
                outputs = []
                for encoded in serialized:
                    batch = collate_serialized([encoded], self.pad_id)
                    batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    outputs.append(self.model(batch, temperature=self.temperature))
                    executed_tokens += batch["input_ids"].numel()
            for question, output in zip(questions, outputs):
                probabilities = output.probabilities[0].cpu().tolist()
                answers[question.decision.id] = format_answer(question, probabilities)
            elapsed = time.perf_counter() - start
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": logical_tokens, "output_tokens": 0,
                          "expanded_input_tokens": expanded_tokens,
                          "executed_input_tokens": executed_tokens,
                          "reused_prefix_tokens": expanded_tokens - logical_tokens if execution == "shared_prefix" else 0},
                "runtime": {"execution": execution, "cache_hit": False,
                            "prefix_reuse": execution == "shared_prefix",
                            "state_prefills": (sum(len(x[0].branches) if x[2] is not None else 1 for x in prepared)
                                               if prepared is not None else 1 if execution == "shared_prefix" else 0),
                            "seconds": elapsed}}


    @torch.inference_mode()
    def predict_decisions(self, decisions, *, images=None):
        """Return aligned probabilities; batch independent text decisions on the GPU.

        The caller bounds the batch by lengths/options. Images are per decision;
        their encoder and state+question are shared across all that decision's options.
        """
        if not decisions:
            return []
        with self.lock:
            if images is not None:
                if len(images) != len(decisions):
                    raise ValueError("images must align with decisions")
                from .multimodal import serialize_images
                result=[]
                for decision, media_values in zip(decisions,images):
                    if not media_values:
                        encoded=self.serializer(decision);media={}
                    else:
                        encoded,media=serialize_images(self,decision,media_values)
                    outputs,_=predict_one_prefix(self.model,encoded,device=self.device,
                        branch_microbatch_size=self.model.branch_microbatch_size,pad_token_id=self.pad_id,
                        temperature=self.temperature,media=media)
                    result.append(outputs[0].probabilities[0].float().cpu().tolist())
                return result
            encoded=[self.serializer(d) for d in decisions]
            if any(x.logical_token_count>self.max_request_tokens for x in encoded):
                raise ValueError("decision exceeds token budget")
            batch=collate_serialized(encoded,self.pad_id)
            batch={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            out=self.model(batch,temperature=self.temperature).probabilities.float().cpu().tolist()
            return [p[:len(d.options)] for p,d in zip(out,decisions)]
