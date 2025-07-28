
# __init__.py
"""
ComfyUI MusicGen Node Package
Hugging Face Transformers-based MusicGen integration for ComfyUI
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import (MusicGenHF, SaveAudioHF, LoadAudioHF)

NODE_CLASS_MAPPINGS = {
    "MusicGenHF": MusicGenHF,
    "Save Audio HF": SaveAudioHF,
    "Load Audio HF": LoadAudioHF,
}

# This is required for ComfyUI to recognize the node package
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Optional: Add version info
__version__ = "1.0.0"
