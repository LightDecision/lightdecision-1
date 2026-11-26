"""Capabilities of the released vision-language backbone."""
from lightdecision.models.base import BackendCapabilities

def capabilities():
    return BackendCapabilities(prefix_reuse=True, fork=True, multi_token_continuation=True)
