"""TypeSafe image assets: positional references and dependency-free multipart input."""
import base64
import binascii
import json
import re
from email import policy
from email.parser import BytesParser

IMAGE_MARKER = '<|vision_start|><|image_pad|><|vision_end|>'


def multipart_request(raw, content_type):
    """Normalize official SDK file parts into the same JSON asset representation."""
    message = BytesParser(policy=policy.default).parsebytes(
        ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + raw)
    if not message.is_multipart() or message.defects:
        raise ValueError('invalid multipart body')
    parts = list(message.iter_parts())
    names = [p.get_param('name', header='content-disposition') for p in parts]
    if not names or names[0] != 'request' or any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError('multipart requires request first and unique nonempty part names')
    if any(p.defects or p.is_multipart() for p in parts):
        raise ValueError('invalid multipart part')
    body = json.loads(parts[0].get_payload(decode=True))
    if not isinstance(body, dict) or not isinstance(body.get('assets', {}), dict):
        raise ValueError('multipart request requires an object and an assets map')
    files = dict(zip(names[1:], parts[1:]))
    used = set()
    for ref, asset in body.get('assets', {}).items():
        if not isinstance(asset, dict):
            raise ValueError('asset must be an object')
        if 'part' not in asset:
            continue
        name = asset['part']
        if not isinstance(name, str) or name not in files or name in used:
            raise ValueError('each image part must belong to exactly one asset')
        part = files[name]
        mime = part.get_content_type() if part.get('Content-Type') else None
        if mime not in (None, 'image/png', 'image/jpg', 'image/jpeg', 'application/octet-stream'):
            raise ValueError('unsupported image part content type')
        data = part.get_payload(decode=True)
        fmt = 'png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'jpg' if data.startswith(b'\xff\xd8\xff') else None
        if fmt is None:
            raise ValueError('image parts must contain PNG or JPEG files')
        declared = asset.get('format', fmt)
        if declared not in ('png', 'jpg', 'jpeg') or ('jpg' if declared == 'jpeg' else declared) != fmt:
            raise ValueError('image part format does not match file')
        # Official precedence: part wins over data_base64.
        body['assets'][ref] = {**asset, 'format': fmt, 'data_base64': base64.b64encode(data).decode('ascii')}
        del body['assets'][ref]['part']
        used.add(name)
    if used != set(files):
        raise ValueError('unreferenced multipart file part')
    return body


class AssetResolver:
    def __init__(self, assets):
        if not isinstance(assets, dict):
            raise ValueError('assets must be an object')
        self.images = {}
        self.used = set()
        for ref, asset in assets.items():
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError('asset references must be nonempty strings')
            if not isinstance(asset, dict) or asset.get('type', 'image') != 'image':
                raise ValueError('assets currently support type image')
            if 'part' in asset:
                raise ValueError('asset part requires multipart/form-data')
            fmt, data = asset.get('format'), asset.get('data_base64')
            if fmt not in ('png', 'jpg', 'jpeg') or not isinstance(data, str) or not data:
                raise ValueError('image assets require format png/jpg/jpeg and data_base64')
            try:
                raw = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError('data_base64 must be standard base64 without a data: prefix or whitespace') from exc
            if base64.b64encode(raw).decode('ascii') != data:
                raise ValueError('data_base64 must use canonical padded base64')
            if not raw.startswith(b'\x89PNG\r\n\x1a\n' if fmt == 'png' else b'\xff\xd8\xff'):
                raise ValueError('asset format does not match image file')
            self.images[ref] = f'data:image/{fmt};base64,{data}'
        # Whole values take precedence; inline ambiguous overlaps use longest at each position.
        self.pattern = re.compile('|'.join(re.escape(k) for k in sorted(self.images, key=len, reverse=True))) if assets else None

    def render(self, value):
        """Replace string values only, retaining structure, keys, order and repeats."""
        images = []

        def replace_ref(ref):
            self.used.add(ref)
            images.append(self.images[ref])
            return IMAGE_MARKER

        def walk(item):
            if isinstance(item, str):
                if item in self.images:
                    return replace_ref(item)
                return self.pattern.sub(lambda m: replace_ref(m.group()), item) if self.pattern else item
            if isinstance(item, dict):
                return {key: walk(item[key]) for key in sorted(item)}
            if isinstance(item, list):
                return [walk(v) for v in item]
            return item

        if not isinstance(value, (str, dict, list)):
            raise ValueError('content must be a string, object or array')
        rendered = walk(value)
        text = rendered if isinstance(rendered, str) else json.dumps(rendered, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return text, tuple(images)

    def finish(self):
        unused = set(self.images) - self.used
        if unused:
            raise ValueError('every asset must be referenced in state, instructions or criteria')
