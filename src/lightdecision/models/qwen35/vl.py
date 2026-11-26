"""Native Qwen3.5 vision-language backbone; the visual tower remains registered."""
from pathlib import Path
import torch
from lightdecision.models.base import ModelBackendSpec
from .adapter import Qwen35Adapter
from .prefix_state import Qwen35PrefixState
from .spec import capabilities

class Qwen35VLAdapter(Qwen35Adapter):
    def __init__(self, backbone, processor):
        super().__init__(backbone)
        self.processor = processor
        self.freeze_vision()

    @property
    def text_model(self):
        return self.backbone.language_model

    @property
    def frozen_visual_module(self):
        return self.backbone.visual

    def freeze_vision(self):
        self.backbone.visual.requires_grad_(False)
        self.backbone.visual.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_vision()
        return self

    def get_hidden_size(self):
        return self.backbone.config.text_config.hidden_size

    @classmethod
    def from_prepared(cls, path, *, device='cpu', dtype='bfloat16', attn_implementation='sdpa', **kwargs):
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        path = Path(path)
        backbone = Qwen3_5Model.from_pretrained(path/'backbone', dtype=dtype,
            attn_implementation=attn_implementation, local_files_only=True)
        processor = AutoProcessor.from_pretrained(path/'processor', local_files_only=True)
        return cls(backbone.to(device), processor)

    def forward_hidden(self, input_ids, attention_mask, position_ids=None, **media):
        # Text inputs use the language-model path with the same weights.
        if not media:
            return self.text_model(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False).last_hidden_state
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False, **media).last_hidden_state

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None, **media):
        if not media:
            return super().prefill(input_ids, attention_mask)
        if self.training:
            raise ValueError('prefix execution requires eval mode')
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError('multimodal prefix must be unpadded')
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, **media)
        return Qwen35PrefixState(result.past_key_values, input_ids.shape[1], input_ids.shape[0],
            self._state_owner, position_delta=result.rope_deltas.clone(),
            last_hidden=result.last_hidden_state[:, -1].clone())




BACKEND_SPEC = ModelBackendSpec('qwen35_vl', 1, Qwen35VLAdapter.from_prepared, capabilities)
