"""Native image preprocessing for Python and HTTP inference."""
from dataclasses import replace
from pathlib import Path
import base64,io
import torch
from PIL import Image
from lightdecision.models.decision.serializer import SerializedBranch

def load_images(values):
    result=[]
    for index,value in enumerate(values):
        try:
            if isinstance(value,Image.Image): image=value
            elif isinstance(value,str) and value.startswith('data:image/'):
                image=Image.open(io.BytesIO(base64.b64decode(value.split(',',1)[1],validate=True)))
            elif isinstance(value,(str,Path)): image=Image.open(value)
            else: raise ValueError('images must be PIL images, local image paths or image data URLs')
            result.append(image.convert('RGB'))
        except (OSError,ValueError,IndexError) as exc:
            raise ValueError(f'cannot read images[{index}]: {exc}') from exc
    return result

def serialize_images(engine, decision, images):
    if not hasattr(engine.model.adapter,'processor'):
        raise ValueError('this artifact does not retain a vision encoder')
    encoded=engine.serializer(decision)
    marker='<|vision_start|><|image_pad|><|vision_end|>'
    text=f'<STATE>{marker*len(images)}{decision.state}</STATE>\n<QUESTION>{decision.question}</QUESTION>\n'
    inputs=engine.model.adapter.processor(text=[text],images=load_images(images),return_tensors='pt',add_special_tokens=False)
    ids=tuple(inputs.pop('input_ids')[0].tolist());inputs.pop('attention_mask',None)
    logical=len(ids)+sum(map(len,encoded.option_token_ids))
    if logical>min(engine.max_request_tokens,engine.serializer.max_decision_tokens):
        raise ValueError('multimodal decision exceeds the logical token budget')
    branches=tuple(SerializedBranch(ids+option,len(ids)-1,len(ids)+len(option)-1) for option in encoded.option_token_ids)
    if any(len(b.input_ids)>engine.serializer.max_branch_length for b in branches):
        raise ValueError('multimodal decision exceeds the branch token budget')
    encoded=replace(encoded,state_ids=ids,question_ids=(),branches=branches,logical_token_count=logical)
    media={k:v.to(engine.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
    return encoded,media


def serialize_asset_question(engine, question, legacy_images=()):
    """Keep image occurrences in their original state/question/option positions.

    Prefix-only images use the existing fast cache path. Image-bearing options
    use complete native visual branches, avoiding incorrect text-only cache RoPE.
    """
    from lightdecision.api.assets import IMAGE_MARKER
    from lightdecision.models.decision.serializer import SerializedDecision
    adapter = engine.model.adapter
    if not hasattr(adapter, 'processor'):
        raise ValueError('this artifact does not retain a vision encoder')
    d = question.decision
    image_cache = {}

    def process(text, images):
        if images:
            loaded = []
            for value in images:
                if value not in image_cache:
                    image_cache[value] = load_images([value])[0]
                loaded.append(image_cache[value])
            inputs = dict(adapter.processor(text=[text], images=loaded, return_tensors='pt', add_special_tokens=False))
            ids = tuple(inputs.pop('input_ids')[0].tolist())
            inputs.pop('attention_mask', None)
            return ids, inputs
        return tuple(engine.tokenizer.encode(text, add_special_tokens=False)), {}

    prefix_images = tuple(legacy_images) + question.prefix_images
    text = f'<STATE>{IMAGE_MARKER * len(legacy_images)}{d.state}</STATE>\n<QUESTION>{d.question}</QUESTION>\n'
    prefix_ids, prefix_media = process(text, prefix_images)
    if engine.serializer.add_bos_token:
        prefix_ids = (engine.tokenizer.bos_token_id,) + prefix_ids
        for k in list(prefix_media):
            if k.endswith('token_type_ids'):
                prefix_media[k] = torch.nn.functional.pad(prefix_media[k], (1, 0))
    option_ids, options_media = [], []
    for option, images in zip(d.options, question.option_images):
        text = f'<OPTION><NAME>{option.name}</NAME>'
        if option.criteria is not None:
            text += f'<CRITERIA>{option.criteria}</CRITERIA>'
        ids, media = process(text + '</OPTION>', images)
        option_ids.append(ids)
        options_media.append(media)
    logical = len(prefix_ids) + sum(map(len, option_ids))
    branches = tuple(SerializedBranch(prefix_ids + ids, len(prefix_ids)-1, len(prefix_ids)+len(ids)-1) for ids in option_ids)
    if logical > min(engine.max_request_tokens, engine.serializer.max_decision_tokens):
        raise ValueError('multimodal decision exceeds the logical token budget; truncation disabled')
    if any(len(b.input_ids) > engine.serializer.max_branch_length for b in branches):
        raise ValueError('multimodal decision exceeds the branch token budget; truncation disabled')
    encoded = SerializedDecision(d.id, tuple(o.id for o in d.options), prefix_ids, (), tuple(option_ids), branches, logical)
    if not any(question.option_images):
        return encoded, prefix_media, None
    branch_media = []
    for ids, media in zip(option_ids, options_media):
        combined = {}
        for k in prefix_media.keys() | media.keys():
            a, b = prefix_media.get(k), media.get(k)
            if k.endswith('token_type_ids'):
                if a is None: a = b.new_zeros((1, len(prefix_ids)))
                if b is None: b = a.new_zeros((1, len(ids)))
                combined[k] = torch.cat((a, b), dim=-1)
            elif a is None: combined[k] = b
            elif b is None: combined[k] = a
            else: combined[k] = torch.cat((a, b), dim=0)
        branch_media.append(combined)
    return encoded, prefix_media, branch_media


@torch.inference_mode()
def predict_asset_branches(engine, encoded, branch_media):
    """Bound activation memory to one visual option branch, then score the whole set."""
    from lightdecision.compute import compute_context
    features, question = [], None
    for branch, media in zip(encoded.branches, branch_media):
        ids = torch.tensor([branch.input_ids], device=engine.device)
        native = {k: v.to(engine.device) if isinstance(v, torch.Tensor) else v for k, v in media.items()}
        with compute_context(engine.compute_contract, 'backbone', engine.device):
            hidden = engine.model.adapter.forward_hidden(ids, attention_mask=None, **native)
        if question is None:
            question = hidden[:, branch.question_end].clone()
        features.append(hidden[0, branch.option_end].clone())
        del hidden, native
    options = torch.stack(features).unsqueeze(0)
    mask = torch.ones((1, len(features)), dtype=torch.bool, device=engine.device)
    return engine.model.score_features(question, options, mask, temperature=engine.temperature), sum(len(b.input_ids) for b in encoded.branches)
