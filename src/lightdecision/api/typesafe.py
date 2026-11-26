"""TypeSafe /v1/systemone JSON mapping to the one decision primitive.

Wire contract: https://docs.typesafe.ai/primitives/{choice,noul,score}.
LightDecision extends it with images and unbounded option/level cardinality.
Question identifiers and Score level indices never enter model text.
"""

from dataclasses import dataclass
import json
import math

from lightdecision.models.decision.schema import Decision, Option
from .assets import AssetResolver


class InvalidDecisionRequest(ValueError):
    """A caller request fails the protocol or token budget contract."""


def structured_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raise ValueError("content must be a string, object or array")


@dataclass(frozen=True)
class TypedQuestion:
    decision: Decision
    primitive: str
    legend: dict | None = None
    prefix_images: tuple[str, ...] = ()
    option_images: tuple[tuple[str, ...], ...] = ()


def parse_request(body, *, model_ids):
    if not isinstance(body, dict) or "state" not in body:
        raise ValueError("request requires state and questions")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise ValueError("model must be a nonempty string")
    if body["model"] not in model_ids:
        raise ValueError("unknown model")
    if "images" in body:
        images = body["images"]
        if not isinstance(images, list) or any(not isinstance(image, str) or not image for image in images):
            raise ValueError("images must be an array of image data URLs or local image paths")
    assets = AssetResolver(body.get("assets", {}))
    state, state_images = assets.render(body["state"])
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    result = []
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, dict):
            raise ValueError("questions require nonempty string IDs and question objects")
        instructions, question_images = assets.render(question.get("instructions"))
        option_images = []

        def description_text(value):
            text, images = assets.render(value)
            option_images.append(images)
            return text

        def optional_description(value):
            if value is None:
                option_images.append(())
                return None
            return description_text(value)
        primitive, criteria = question.get("type"), question.get("criteria")
        legend = None
        if primitive == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("choice criteria must be a nonempty map of option names to descriptions")
            options = tuple(Option(name, name, optional_description(description))
                            for name, description in criteria.items())
        elif primitive == "noul":
            criteria = {} if criteria is None else criteria
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise ValueError("noul criteria supports only false and true descriptions")
            options = tuple(Option(key, name, optional_description(criteria.get(key)))
                            for key, name in (("false", "False"), ("true", "True")))
        elif primitive == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("score criteria requires a nonempty array of level descriptions")
            options = tuple(Option(str(index), description_text(description)) for index, description in enumerate(criteria))
            legend = {str(index): description for index, description in enumerate(criteria)}
        else:
            raise ValueError("question type must be choice, noul or score")
        decision = Decision(key, state, instructions, options,
                            tuple(float(i) for i in range(len(options))) if primitive == "score" else None)
        result.append(TypedQuestion(decision, primitive, legend, state_images + question_images, tuple(option_images)))
    assets.finish()
    return result


def format_answer(question, probabilities):
    probabilities = [float(p) for p in probabilities]
    if len(probabilities) != len(question.decision.options):
        raise RuntimeError("model returned an incomplete probability vector")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=2e-6):
        raise RuntimeError("model returned an invalid probability distribution")
    if question.primitive == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    count = len(probabilities)
    mode = max(range(count), key=probabilities.__getitem__)
    # TypeSafe's documented statistics, not a temperature or a change to p.
    # A single possible outcome is the degenerate confidence=1 extension.
    if count == 1:
        confidence = 1.0
    elif question.primitive == "choice":
        confidence = max(0.0, min(1.0, (probabilities[mode] - 1 / count) / (1 - 1 / count)))
    else:
        spread = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
        even_spread = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
        confidence = max(0.0, 1 - spread / even_spread)
    answer = {"type": question.primitive,
              "probabilities": {option.id: p for option, p in zip(question.decision.options, probabilities)},
              "confidence": confidence}
    if question.primitive == "choice":
        answer["choice"] = question.decision.options[mode].name
    else:
        answer.update(score=sum(i * p for i, p in enumerate(probabilities)), legend=question.legend)
    return answer
