"""
ComfyUI MusicGen Nodes Package
"""

# Import all node classes from submodules
from .musicgen_node import HuggingFaceMusicGen
from .bpm_nodes import BPMDurationInput
from .preview_nodes import LoopingAudioPreview, SmoothAudioQueue
from .advanced_loop_nodes import ProfessionalLoopTransition
from .io_nodes import SaveAudioStandalone, LoadAudioStandalone, MusicGenAudioToFile

# Collect all node mappings
NODE_CLASS_MAPPINGS = {
    "HuggingFaceMusicGen": HuggingFaceMusicGen,
    "BPMDurationInput": BPMDurationInput,
    "LoopingAudioPreview": LoopingAudioPreview,
    "SmoothAudioQueue": SmoothAudioQueue,
    "ProfessionalLoopTransition": ProfessionalLoopTransition,
    "MusicGenAudioToFile": MusicGenAudioToFile,
    "SaveAudioStandalone": SaveAudioStandalone,
    "LoadAudioStandalone": LoadAudioStandalone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceMusicGen": "MusicGen (Hugging Face)",
    "BPMDurationInput": "BPM Duration Calculator",
    "LoopingAudioPreview": "Looping Audio Preview",
    "SmoothAudioQueue": "Smooth Audio Queue",
    "ProfessionalLoopTransition": "Professional Loop Transition",
    "MusicGenAudioToFile": "Save MusicGen Audio (Legacy)",
    "SaveAudioStandalone": "Save Audio (WAV/FLAC/MP3/Opus)",
    "LoadAudioStandalone": "Load Audio File"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']