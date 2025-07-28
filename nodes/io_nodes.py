"""
Audio input/output nodes for loading and saving
"""

import os
from ..utils.audio_utils import save_audio_standalone, load_audio_file


class LoadAudioStandalone:
    """Standalone audio loader that doesn't depend on ComfyUI's LoadAudio node"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio_path": ("STRING", {"default": "", "multiline": False})}}
    
    CATEGORY = "audio/musicgen"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    
    def load(self, audio_path):
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        waveform, sample_rate = load_audio_file(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio,)


class SaveAudioStandalone:
    """Standalone audio saver with multiple format support"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "musicgen_audio"}),
                "format": (["wav", "flac", "mp3", "opus"], {"default": "wav"}),
                "quality": (["128k", "192k", "320k", "V0"], {"default": "128k"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = True
    
    def save_audio(self, audio, filename_prefix="musicgen_audio", format="wav", quality="128k"):
        filepath = save_audio_standalone(audio, filename_prefix, format, quality=quality)
        return (filepath,)


class MusicGenAudioToFile:
    """Simplified audio saver (deprecated - use SaveAudioStandalone instead)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {
                    "default": "musicgen_output",
                    "multiline": False
                }),
                "format": (["wav", "flac", "mp3", "opus"], {
                    "default": "wav"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = True
    
    def save_audio(self, audio, filename, format):
        filepath = save_audio_standalone(audio, filename, format)
        return (filepath,)