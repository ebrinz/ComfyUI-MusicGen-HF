
# __init__.py
"""
ComfyUI MusicGen Node Package
Hugging Face Transformers-based MusicGen integration for ComfyUI
"""

# Import from the modular node structure
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# This is required for ComfyUI to recognize the node package
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.1.0"
