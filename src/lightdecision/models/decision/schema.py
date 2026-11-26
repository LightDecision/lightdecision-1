"""Public decision records and tensor outputs, without a torch import at load time."""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class Option:
    id: str
    name: str
    criteria: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("option id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("option name must be a non-empty string")
        if self.criteria is not None and not isinstance(self.criteria, str):
            raise TypeError("option criteria must be text or None")


@dataclass(frozen=True)
class ImageReference:
    path: str
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("image path must be nonempty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("image reference requires a lowercase SHA-256")


@dataclass(frozen=True)
class Decision:
    id: str
    state: str
    question: str
    options: tuple[Option, ...]
    score_values: tuple[float, ...] | None = None
    images: tuple[ImageReference, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("decision id must be a non-empty string")
        if not isinstance(self.state, str) or not isinstance(self.question, str):
            raise TypeError("state and question must be serialized text")
        object.__setattr__(self, "images", tuple(self.images))
        if not all(isinstance(image, ImageReference) for image in self.images):
            raise TypeError("images must contain ImageReference instances")
        object.__setattr__(self, "options", tuple(self.options))
        if not self.options:
            raise ValueError("a decision requires at least one option")
        if not all(isinstance(option, Option) for option in self.options):
            raise TypeError("options must contain Option instances")
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("option ids must be unique within a decision")
        if self.score_values is not None:
            values = tuple(self.score_values)
            if len(values) != len(self.options):
                raise ValueError("score_values must align with options")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("score_values must be finite numbers")
            object.__setattr__(self, "score_values", values)


@dataclass(frozen=True)
class DecisionFeatures:
    question_features: "Tensor"
    option_features: "Tensor"
    option_mask: "Tensor"


@dataclass(frozen=True)
class DecisionOutput:
    logits: "Tensor"
    log_probs: "Tensor"
    probabilities: "Tensor"
    option_mask: "Tensor"
