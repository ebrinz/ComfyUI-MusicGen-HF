"""
Audio conditioning queue nodes for loading output audio and using it as conditioning input
"""

import torch
import os
import time
import glob
import hashlib

try:
    from ..utils.audio_utils import load_audio_file
except ImportError:
    # Fallback for direct testing
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.audio_utils import load_audio_file

try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


class AudioOutputToConditioningQueue:
    """
    Node that captures audio output and queues it for conditioning the next generation
    Can use either direct audio input or load from files
    Enables audio continuation workflows
    """
    
    # Class-level queue system for conditioning audio
    _conditioning_queue = {}
    _last_generated_audio = {}
    
    @classmethod
    def store_generated_audio(cls, queue_id, audio_dict):
        """Store generated audio for later use as conditioning"""
        cls._last_generated_audio[queue_id] = audio_dict
        print(f"🎵 Stored generated audio for queue: {queue_id}")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "queue_id": ("STRING", {
                    "default": "main_chain",
                    "tooltip": "Unique identifier for this conditioning chain"
                }),
                "source_mode": (["last_generated", "load_file", "direct_input"], {
                    "default": "last_generated",
                    "tooltip": "Source of conditioning audio"
                }),
                "max_conditioning_duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "Maximum duration for conditioning audio"
                })
            },
            "optional": {
                "audio_input": ("AUDIO", {
                    "tooltip": "Direct audio input (for direct_input mode)"
                }),
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to audio file (for load_file mode)"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("conditioning_audio", "queue_info", "loaded_file")
    FUNCTION = "load_and_queue_audio"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = False
    
    def get_output_directory(self):
        """Get the ComfyUI output directory"""
        if COMFY_AVAILABLE:
            try:
                output_dir = folder_paths.get_output_directory()
                audio_dir = os.path.join(output_dir, "audio")
                return audio_dir
            except:
                pass
        
        # Fallback
        return os.path.expanduser("~/ComfyUI_output/audio")
    
    def get_audio_files(self, directory, format_filter="all"):
        """Get list of audio files in directory, sorted by modification time"""
        if not os.path.exists(directory):
            return []
        
        # Define supported formats
        formats = {
            "all": ["*.wav", "*.mp3", "*.flac", "*.opus", "*.m4a", "*.aac"],
            "wav": ["*.wav"],
            "mp3": ["*.mp3"], 
            "flac": ["*.flac"],
            "opus": ["*.opus"]
        }
        
        patterns = formats.get(format_filter, formats["all"])
        
        audio_files = []
        for pattern in patterns:
            audio_files.extend(glob.glob(os.path.join(directory, pattern)))
        
        # Sort by modification time (newest first)
        audio_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return audio_files
    
    def trim_audio_duration(self, audio_dict, max_duration):
        """Trim audio to maximum duration for conditioning"""
        if max_duration <= 0:
            return audio_dict
        
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        max_samples = int(max_duration * sample_rate)
        current_samples = waveform.shape[-1]
        
        if current_samples > max_samples:
            # Trim from the end (keep the beginning for conditioning)
            trimmed_waveform = waveform[..., :max_samples]
            return {
                "waveform": trimmed_waveform,
                "sample_rate": sample_rate
            }
        
        return audio_dict
    
    def load_and_queue_audio(self, queue_id, source_mode, max_conditioning_duration, 
                           audio_input=None, file_path=""):
        
        try:
            # Initialize queue if needed
            if queue_id not in self._conditioning_queue:
                self._conditioning_queue[queue_id] = []
            
            queue = self._conditioning_queue[queue_id]
            loaded_file = "none"
            
            # Process based on source mode
            if source_mode == "direct_input" and audio_input is not None:
                # Use direct audio input
                print(f"🎵 Using direct audio input for conditioning")
                
                # Ensure proper format
                if isinstance(audio_input, dict) and "waveform" in audio_input:
                    conditioning_audio = audio_input
                else:
                    return (
                        {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
                        "Invalid direct audio input format",
                        "error"
                    )
                
                loaded_file = "direct_input"
                
            elif source_mode == "load_file" and file_path and file_path.strip():
                # Load from specified file path
                target_file = file_path.strip()
                
                # If it's not an absolute path, try to find it in ComfyUI directories
                if not os.path.isabs(target_file):
                    # Try output directory first
                    output_dir = self.get_output_directory()
                    abs_target = os.path.join(output_dir, target_file)
                    if os.path.exists(abs_target):
                        target_file = abs_target
                    else:
                        # Try input directory
                        try:
                            if COMFY_AVAILABLE:
                                input_dir = folder_paths.get_input_directory()
                                abs_target = os.path.join(input_dir, target_file)
                                if os.path.exists(abs_target):
                                    target_file = abs_target
                        except:
                            pass
                
                if not os.path.exists(target_file):
                    return (
                        {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
                        f"File not found: {file_path}",
                        "not_found"
                    )
                
                print(f"🎵 Loading conditioning audio from file: {os.path.basename(target_file)}")
                
                try:
                    # Load audio using utility function
                    waveform, sample_rate = load_audio_file(target_file)
                    
                    # Create audio dict
                    conditioning_audio = {
                        "waveform": waveform.unsqueeze(0) if waveform.dim() == 2 else waveform,
                        "sample_rate": sample_rate
                    }
                    
                    loaded_file = os.path.basename(target_file)
                    
                except Exception as e:
                    return (
                        {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
                        f"Error loading file {target_file}: {str(e)}",
                        "error"
                    )
                    
            elif source_mode == "last_generated":
                # Use the most recent generated audio from the queue or last generated audio store
                print(f"🎵 Using last generated audio for conditioning")
                
                # Check if we have stored last generated audio for this queue
                if queue_id in self._last_generated_audio and self._last_generated_audio[queue_id]:
                    conditioning_audio = self._last_generated_audio[queue_id]
                    loaded_file = "last_generated"
                elif queue:
                    # Fall back to most recent in queue
                    conditioning_audio = queue[-1]["audio"]
                    loaded_file = os.path.basename(queue[-1]["file_path"])
                    print(f"🔄 Using most recent queued audio: {loaded_file}")
                else:
                    # Try to find the most recent audio file in output directory
                    output_dir = self.get_output_directory()
                    if os.path.exists(output_dir):
                        audio_files = self.get_audio_files(output_dir, "all")
                        if audio_files:
                            latest_file = audio_files[0]  # Already sorted by modification time
                            try:
                                waveform, sample_rate = load_audio_file(latest_file)
                                conditioning_audio = {
                                    "waveform": waveform.unsqueeze(0) if waveform.dim() == 2 else waveform,
                                    "sample_rate": sample_rate
                                }
                                loaded_file = os.path.basename(latest_file)
                                print(f"📁 Found and loaded latest audio file: {loaded_file}")
                            except Exception as e:
                                print(f"❌ Error loading latest file {latest_file}: {e}")
                                conditioning_audio = None
                        else:
                            conditioning_audio = None
                    else:
                        conditioning_audio = None
                
                if conditioning_audio is None:
                    return (
                        {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
                        "No last generated audio available",
                        "none"
                    )
            else:
                return (
                    {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
                    f"Invalid source mode or missing required parameters: {source_mode}",
                    "error"
                )
            
            # Trim to max conditioning duration
            conditioning_audio = self.trim_audio_duration(conditioning_audio, max_conditioning_duration)
            
            # Add to queue if it's not already a direct input
            if source_mode != "direct_input":
                queue_entry = {
                    "audio": conditioning_audio,
                    "source": source_mode,
                    "loaded_at": time.time(),
                    "duration": conditioning_audio["waveform"].shape[-1] / conditioning_audio["sample_rate"],
                    "file_path": file_path if source_mode == "load_file" else "generated"
                }
                
                queue.append(queue_entry)
                
                # Maintain queue length limit (keep last 10)
                while len(queue) > 10:
                    removed = queue.pop(0)
                    print(f"🗑️ Removed old audio from queue")
            
            # Create queue info
            queue_info = f"Conditioning Queue ({queue_id}):\n"
            queue_info += f"Source mode: {source_mode}\n"
            queue_info += f"Queue length: {len(queue)}/10\n"
            queue_info += f"Max duration: {max_conditioning_duration:.1f}s\n"
            queue_info += f"Loaded: {loaded_file}\n"
            
            if conditioning_audio:
                duration = conditioning_audio["waveform"].shape[-1] / conditioning_audio["sample_rate"]
                queue_info += f"Audio duration: {duration:.2f}s\n"
                queue_info += f"Sample rate: {conditioning_audio['sample_rate']}Hz\n"
            
            if queue and len(queue) > 0:
                queue_info += f"\nRecent queue entries:\n"
                for i, entry in enumerate(reversed(queue[-3:])):  # Show last 3
                    source_info = entry.get("source", "unknown")
                    duration = entry["duration"]
                    queue_info += f"  {i+1}. {source_info} ({duration:.1f}s)\n"
            
            print(f"✅ Conditioning audio ready: {loaded_file} ({conditioning_audio['waveform'].shape[-1] / conditioning_audio['sample_rate']:.2f}s)")
            
            return (conditioning_audio, queue_info, loaded_file)
            
        except Exception as e:
            error_msg = f"Error in audio conditioning queue: {str(e)}"
            print(f"❌ {error_msg}")
            return (
                {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
                error_msg,
                "error"
            )


class ConditioningQueueManager:
    """
    Node for managing and viewing conditioning queues
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "queue_id": ("STRING", {
                    "default": "main_chain",
                    "tooltip": "Queue identifier to manage"
                }),
                "action": (["view", "clear", "remove_oldest", "remove_newest"], {
                    "default": "view",
                    "tooltip": "Action to perform on the queue"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("queue_status",)
    FUNCTION = "manage_queue"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = True
    
    def manage_queue(self, queue_id, action):
        try:
            # Access the queue from the other node
            if queue_id not in AudioOutputToConditioningQueue._conditioning_queue:
                return (f"Queue '{queue_id}' not found. Create it first with AudioOutputToConditioningQueue node.",)
            
            queue = AudioOutputToConditioningQueue._conditioning_queue[queue_id]
            
            if action == "clear":
                queue.clear()
                status = f"✅ Cleared queue '{queue_id}'"
                
            elif action == "remove_oldest" and queue:
                removed = queue.pop(0)
                filename = os.path.basename(removed["file_path"])
                status = f"🗑️ Removed oldest: {filename}"
                
            elif action == "remove_newest" and queue:
                removed = queue.pop()
                filename = os.path.basename(removed["file_path"])
                status = f"🗑️ Removed newest: {filename}"
                
            else:  # view
                status = f"Queue '{queue_id}' Status:\n"
                status += f"Length: {len(queue)} items\n"
                
                if queue:
                    status += "Files in queue:\n"
                    for i, entry in enumerate(queue):
                        filename = os.path.basename(entry["file_path"])
                        duration = entry["duration"]
                        loaded_time = time.strftime("%H:%M:%S", time.localtime(entry["loaded_at"]))
                        status += f"  {i+1}. {filename} ({duration:.1f}s) - loaded at {loaded_time}\n"
                else:
                    status += "Queue is empty"
            
            return (status,)
            
        except Exception as e:
            error_msg = f"Error managing queue: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)