"""Select verified upstream kernels without replacing their implementation."""
import importlib.metadata

def configure_upstream_fla():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    if importlib.metadata.version('fla-core') != '0.5.2' or qwen.chunk_gated_delta_rule is None:
        raise RuntimeError('Install fla-core==0.5.2 in the inference environment')
    # Keep norm and convolution unchanged while validating the delta path independently.
    qwen.FusedRMSNormGated = None
    return {'delta': 'fla-core-0.5.2', 'norm': 'transformers', 'conv': 'torch'}
