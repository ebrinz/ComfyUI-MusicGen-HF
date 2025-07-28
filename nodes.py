import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import tempfile
import os
import folder_paths
import av
import io
import json
import hashlib
from comfy.cli_args import args

class HuggingFaceMusicGen:
    """
    ComfyUI Node for Facebook's MusicGen via Hugging Face Transformers
    Supports CUDA, MPS (Apple Silicon), and CPU
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_size": (["small", "medium", "large"], {
                    "default": "small"
                }),
                "duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.00001
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True
                }),
                "max_new_tokens": ("INT", {
                    "default": 256,
                    "min": 50,
                    "max": 1503,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 999999999
                })
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "upbeat electronic music with drums and synth"
                }),
                "conditioning_audio": ("AUDIO",),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "duration_override": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "Override duration parameter with value from BPMDurationInput node"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate_audio"
    CATEGORY = "audio/musicgen"
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.current_model_size = None
        self.device = self._get_optimal_device()
        try:
            print(f"MusicGen will use device: {self.device}")
        except OSError:
            pass
    
    def _get_optimal_device(self):
        """Determine the best available device with MPS support"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _move_to_device(self, tensor_or_dict):
        """Safely move tensors to device, handling MPS limitations"""
        if isinstance(tensor_or_dict, dict):
            return {k: self._move_to_device(v) for k, v in tensor_or_dict.items()}
        elif isinstance(tensor_or_dict, torch.Tensor):
            if self.device == "mps":
                # MPS has some dtype limitations, ensure float32 and contiguous
                if tensor_or_dict.dtype == torch.float64:
                    tensor_or_dict = tensor_or_dict.float()
                # Ensure tensor is contiguous before moving to MPS
                tensor_or_dict = tensor_or_dict.contiguous()
                # Move to MPS and immediately allocate storage
                return tensor_or_dict.to(self.device, non_blocking=False)
            return tensor_or_dict.to(self.device)
        else:
            return tensor_or_dict
    
    def load_model(self, model_size):
        """Load the specified model if not already loaded"""
        if self.model is None or self.current_model_size != model_size:
            print(f"Loading MusicGen-{model_size} on {self.device}...")
            
            model_name = f"facebook/musicgen-{model_size}"
            
            try:
                # Load processor (CPU only, no device needed)
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                # Load model with device-specific settings
                if self.device == "mps":
                    # For MPS, use float32 and specific settings
                    self.model = MusicgenForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False
                    )
                elif self.device == "cuda":
                    # For CUDA, can use mixed precision
                    self.model = MusicgenForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=False
                    )
                else:
                    # CPU
                    self.model = MusicgenForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False
                    )
                
                # Move model to device - ensures all params & buffers are on target device
                self.model = self.model.to(self.device, dtype=None)
                self.model.eval()  # Set to evaluation mode
                self.current_model_size = model_size
                
                # Verify all parameters and buffers are on the correct device
                for name, buf in self.model.named_buffers():
                    if buf.device.type != self.device:
                        print(f"⚠️ Warning: {name} buffer still on {buf.device}, expected {self.device}")
                
                print(f"✅ MusicGen-{model_size} loaded successfully on {self.device}")
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                # Fallback to CPU if device loading fails
                if self.device != "cpu":
                    print("Falling back to CPU...")
                    self.device = "cpu"
                    self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
                    self.model = self.model.to("cpu")
                else:
                    raise e
    
    def generate_audio(self, model_size, duration, guidance_scale, do_sample, max_new_tokens, seed, 
                      prompt="upbeat electronic music with drums and synth", conditioning_audio=None, temperature=1.0, duration_override=0.0):
        
        # Load model
        self.load_model(model_size)
        
        # Use duration_override if provided (> 0), otherwise use duration parameter
        effective_duration = duration_override if duration_override > 0.0 else duration
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have manual_seed, but the global seed affects it
            pass
        
        try:
            # Prepare inputs
            if conditioning_audio is not None:
                # Audio-prompted generation
                if isinstance(conditioning_audio, np.ndarray):
                    conditioning_audio = conditioning_audio.astype(np.float32)
                
                inputs = self.processor(
                    audio=conditioning_audio,
                    sampling_rate=self.model.config.audio_encoder.sampling_rate,
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                )
            else:
                # Text-only generation
                inputs = self.processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                )
            
            # Move inputs to device - ensure all tensors are on same device as model
            inputs = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                      for k, v in inputs.items()}
            
            # Calculate max_new_tokens based on effective duration
            # MusicGen generates approximately 50 tokens per second at 32kHz
            calculated_tokens = int(effective_duration * 50)
            # Use the calculated tokens for duration, but respect the model's absolute maximum
            tokens_to_use = min(calculated_tokens, 1503)  # 1503 is the model's hard limit
            
            # Generate audio with proper device handling
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    guidance_scale=guidance_scale,
                    max_new_tokens=tokens_to_use,
                    temperature=temperature if do_sample else 1.0,
                    pad_token_id=self.model.generation_config.pad_token_id,
                )
            
            # Convert to numpy and get sampling rate
            sampling_rate = self.model.config.audio_encoder.sampling_rate
            
            # Handle different audio shapes (mono vs stereo)
            if len(audio_values.shape) == 3:  # (batch, channels, samples)
                audio_data = audio_values[0, 0].cpu().numpy()  # First batch, first channel
            else:  # (batch, samples)
                audio_data = audio_values[0].cpu().numpy()
            
            # Ensure float32 for ComfyUI compatibility
            audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95
            
            # Create ComfyUI AUDIO format
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)  # Add batch dimension
            if len(audio_tensor.shape) == 2:  # (batch, samples)
                audio_tensor = audio_tensor.unsqueeze(1)  # Add channel dimension -> (batch, channels, samples)
            
            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": sampling_rate
            }
            
            # Create info string
            actual_duration = len(audio_data) / sampling_rate
            duration_source = "BPM input" if duration_override > 0.0 else "manual"
            info = f"Generated {actual_duration:.1f}s audio using MusicGen-{model_size} on {self.device}\n"
            info += f"Duration source: {duration_source} ({effective_duration:.2f}s requested)\n"
            info += f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
            info += f"Settings: guidance={guidance_scale}, tokens={tokens_to_use}, seed={seed}\n"
            info += f"Sample rate: {sampling_rate}Hz"
            
            return (audio_output, info)
            
        except Exception as e:
            error_msg = f"Error generating audio: {str(e)}"
            print(error_msg)
            # Return silence on error in ComfyUI AUDIO format
            empty_audio = torch.zeros([1, 1, int(32000 * effective_duration)], dtype=torch.float32)  # 32kHz silence
            empty_audio_output = {
                "waveform": empty_audio,
                "sample_rate": 32000
            }
            return (empty_audio_output, error_msg)


# Standalone audio saving utilities (adapted from ComfyUI's comfy_extras/nodes_audio.py)
def save_audio_standalone(audio, filename_prefix="ComfyUI", format="wav", output_dir=None, quality="128k"):
    """
    Standalone audio saving function that doesn't depend on ComfyUI's SaveAudio nodes
    """
    if output_dir is None:
        try:
            output_dir = folder_paths.get_output_directory()
        except:
            output_dir = os.path.join(os.path.dirname(__file__), "output")
    
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, "audio")
    
    # Generate filename with timestamp
    import time
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.{format}"
    output_path = os.path.join(output_dir, filename)
    
    # Prepare metadata
    metadata = {}
    
    # Opus supported sample rates
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000]
    
    waveform = audio["waveform"].cpu()
    sample_rate = audio["sample_rate"]
    
    for batch_number, batch_waveform in enumerate(waveform):
        if batch_number > 0:
            batch_filename = f"{filename_prefix}_{timestamp}_{batch_number}.{format}"
            batch_output_path = os.path.join(output_dir, batch_filename)
        else:
            batch_output_path = output_path
        
        # Handle Opus sample rate requirements
        current_sample_rate = sample_rate
        current_waveform = batch_waveform
        
        if format == "opus":
            if sample_rate > 48000:
                current_sample_rate = 48000
            elif sample_rate not in OPUS_RATES:
                for rate in sorted(OPUS_RATES):
                    if rate > sample_rate:
                        current_sample_rate = rate
                        break
                if current_sample_rate not in OPUS_RATES:
                    current_sample_rate = 48000
            
            if current_sample_rate != sample_rate:
                current_waveform = torchaudio.functional.resample(batch_waveform, sample_rate, current_sample_rate)
        
        # Create output with specified format
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format=format)
        
        # Set metadata on the container
        for key, value in metadata.items():
            output_container.metadata[key] = value
        
        # Set up the output stream
        if format == "opus":
            out_stream = output_container.add_stream("libopus", rate=current_sample_rate)
            if quality == "64k":
                out_stream.bit_rate = 64000
            elif quality == "96k":
                out_stream.bit_rate = 96000
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "192k":
                out_stream.bit_rate = 192000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        elif format == "mp3":
            out_stream = output_container.add_stream("libmp3lame", rate=current_sample_rate)
            if quality == "V0":
                out_stream.codec_context.qscale = 1
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        elif format == "flac":
            out_stream = output_container.add_stream("flac", rate=current_sample_rate)
        else:  # wav
            out_stream = output_container.add_stream("pcm_s16le", rate=current_sample_rate)
        
        # Prepare audio frame
        if len(current_waveform.shape) == 1:
            current_waveform = current_waveform.unsqueeze(0)
        
        frame = av.AudioFrame.from_ndarray(
            current_waveform.movedim(0, 1).reshape(1, -1).float().numpy(), 
            format='flt', 
            layout='mono' if current_waveform.shape[0] == 1 else 'stereo'
        )
        frame.sample_rate = current_sample_rate
        frame.pts = 0
        output_container.mux(out_stream.encode(frame))
        
        # Flush encoder
        output_container.mux(out_stream.encode(None))
        output_container.close()
        
        # Write the output to file
        output_buffer.seek(0)
        with open(batch_output_path, 'wb') as f:
            f.write(output_buffer.getbuffer())
        
        print(f"✅ Audio saved: {batch_output_path}")
    
    return output_path

class BPMDurationInput:
    """
    Node that calculates duration based on BPM and beats/measures
    Provides more musical control over timing
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bpm": ("FLOAT", {
                    "default": 120.0,
                    "min": 60.0,
                    "max": 200.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "beats": ("FLOAT", {
                    "default": 16.0,
                    "min": 1.0,
                    "max": 64.0,
                    "step": 0.25,
                    "display": "number"
                }),
                "time_signature": (["4/4", "3/4", "6/8", "2/4"], {
                    "default": "4/4"
                })
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("duration", "info")
    FUNCTION = "calculate_duration"
    CATEGORY = "audio/musicgen"
    
    def calculate_duration(self, bpm, beats, time_signature):
        # Parse time signature
        numerator, denominator = map(int, time_signature.split('/'))
        
        # Calculate beats per measure
        beats_per_measure = numerator
        
        # Calculate duration in seconds
        # Duration = (beats / bpm) * 60 seconds per minute
        duration_seconds = (beats / bpm) * 60.0
        
        # Calculate equivalent tokens (50 tokens per second)
        calculated_tokens = int(duration_seconds * 50)
        max_tokens = 1503
        
        # If calculated tokens exceed max, adjust duration to fit within token limit
        if calculated_tokens > max_tokens:
            max_duration = max_tokens / 50.0
            duration_seconds = max_duration
            actual_beats = (max_duration * bpm) / 60.0
            info = f"Duration capped at {max_duration:.2f}s (max tokens: {max_tokens})\n"
            info += f"Actual beats: {actual_beats:.2f} at {bpm} BPM\n"
            info += f"Time signature: {time_signature}"
        else:
            measures = beats / beats_per_measure
            info = f"Duration: {duration_seconds:.2f}s ({beats} beats at {bpm} BPM)\n"
            info += f"Measures: {measures:.2f} in {time_signature}\n"
            info += f"Estimated tokens: {calculated_tokens}"
        
        return (duration_seconds, info)

class LoopingAudioPreview:
    """
    Audio preview node that loops the audio for continuous playback
    Also provides pass-through output for conditioning audio
    Supports queuing to wait for current playback to finish
    """
    
    # Class-level state for tracking playback timing
    _playback_state = {}
    _audio_queue = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "loop_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of times to loop the audio for preview"
                }),
                "enable_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable audio preview playback"
                }),
                "queue_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Wait for current audio to finish before playing new audio"
                }),
                "instance_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Unique identifier for this preview instance (for multiple nodes)"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("looped_preview", "conditioning_audio", "info")
    FUNCTION = "create_looping_preview"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = True
    
    def create_looping_preview(self, audio, loop_count, enable_preview, queue_mode, instance_id):
        import time
        import hashlib
        
        try:
            # Extract waveform and sample rate
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Calculate durations
            original_duration = waveform.shape[-1] / sample_rate
            looped_duration = original_duration * loop_count
            
            # Create a hash of the audio data to detect changes
            audio_hash = hashlib.md5(waveform.cpu().numpy().tobytes()).hexdigest()
            current_time = time.time()
            
            # Handle queue mode logic
            if queue_mode and enable_preview:
                # Check if this instance has state
                if instance_id not in self._playback_state:
                    self._playback_state[instance_id] = {"end_time": 0, "last_hash": None}
                
                state = self._playback_state[instance_id]
                
                # Check if audio has changed
                audio_changed = state["last_hash"] != audio_hash
                playback_finished = current_time >= state["end_time"]
                
                if audio_changed:
                    if playback_finished:
                        # Start playing new audio immediately
                        state["end_time"] = current_time + looped_duration
                        state["last_hash"] = audio_hash
                        should_play = True
                        queue_status = "Playing new audio"
                    else:
                        # Queue the new audio
                        self._audio_queue[instance_id] = {
                            "audio": audio,
                            "loop_count": loop_count,
                            "hash": audio_hash,
                            "queued_at": current_time
                        }
                        should_play = False
                        remaining_time = state["end_time"] - current_time
                        queue_status = f"Queued (waiting {remaining_time:.1f}s)"
                else:
                    # Same audio, check if we should start queued audio
                    if playback_finished and instance_id in self._audio_queue:
                        queued = self._audio_queue[instance_id]
                        audio = queued["audio"]
                        loop_count = queued["loop_count"]
                        audio_hash = queued["hash"]
                        del self._audio_queue[instance_id]
                        
                        # Update waveform from queued audio
                        waveform = audio["waveform"]
                        sample_rate = audio["sample_rate"]
                        original_duration = waveform.shape[-1] / sample_rate
                        looped_duration = original_duration * loop_count
                        
                        state["end_time"] = current_time + looped_duration
                        state["last_hash"] = audio_hash
                        should_play = True
                        queue_status = "Playing queued audio"
                    else:
                        should_play = not audio_changed
                        queue_status = "Same audio" if should_play else "Waiting for current to finish"
            else:
                # No queue mode - always play immediately
                should_play = True
                queue_status = "Queue mode disabled"
            
            # Create looped version for preview
            if should_play and loop_count > 1:
                looped_waveform = waveform.repeat(1, 1, loop_count)
            else:
                looped_waveform = waveform
            
            # Create looped audio output
            looped_audio = {
                "waveform": looped_waveform if should_play else torch.zeros_like(waveform),
                "sample_rate": sample_rate
            }
            
            # Pass-through original audio for conditioning (always available)
            conditioning_audio = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            # Create info string
            info = f"Audio Preview Info:\n"
            info += f"Original duration: {original_duration:.2f}s\n"
            info += f"Looped duration: {looped_duration:.2f}s ({loop_count}x loops)\n"
            info += f"Sample rate: {sample_rate}Hz\n"
            info += f"Preview enabled: {enable_preview}\n"
            info += f"Queue mode: {queue_mode}\n"
            info += f"Status: {queue_status}\n"
            info += f"Instance: {instance_id}\n"
            info += f"Shape: {list(waveform.shape)}"
            
            return (looped_audio, conditioning_audio, info)
            
        except Exception as e:
            error_msg = f"Error in looping preview: {str(e)}"
            print(error_msg)
            
            # Return original audio on error
            return (audio, audio, error_msg)

class SmoothAudioQueue:
    """
    Node for creating smooth transitions between audio clips
    Supports crossfading and precise duration control
    """
    
    # Class-level audio buffer for smooth playback
    _audio_buffer = {}
    _current_playback = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.01,
                    "tooltip": "Precise target duration (0 = use original length)"
                }),
                "crossfade_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Crossfade duration between clips"
                }),
                "queue_id": ("STRING", {
                    "default": "main",
                    "tooltip": "Queue identifier for multiple audio streams"
                }),
                "auto_play": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically play when audio changes"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("queued_audio", "conditioning_audio", "info")
    FUNCTION = "process_audio_queue"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = True
    
    def validate_duration(self, duration):
        """Round duration to token boundary for precise generation"""
        if duration <= 0:
            return duration
        
        # MusicGen: ~50 tokens per second, max 1503 tokens
        token_duration = 1.0 / 50.0  # ~0.02s per token
        rounded_duration = round(duration / token_duration) * token_duration
        return min(rounded_duration, 30.0)  # cap at 30s max
    
    def trim_to_exact_duration(self, audio_dict, target_duration):
        """Trim audio to exact duration with post-processing"""
        if target_duration <= 0:
            return audio_dict
        
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        samples_needed = int(target_duration * sample_rate)
        current_samples = waveform.shape[-1]
        
        if samples_needed < current_samples:
            # Trim excess
            trimmed_waveform = waveform[..., :samples_needed]
        elif samples_needed > current_samples:
            # Pad with silence if needed
            padding = samples_needed - current_samples
            trimmed_waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            trimmed_waveform = waveform
        
        return {
            "waveform": trimmed_waveform,
            "sample_rate": sample_rate
        }
    
    def create_crossfade(self, audio1, audio2, fade_duration):
        """Create smooth crossfade between two audio clips"""
        if fade_duration <= 0:
            return torch.cat([audio1["waveform"], audio2["waveform"]], dim=-1)
        
        sample_rate = audio1["sample_rate"]
        fade_samples = int(fade_duration * sample_rate)
        
        waveform1 = audio1["waveform"]
        waveform2 = audio2["waveform"]
        
        # Ensure we don't exceed audio lengths
        fade_samples = min(fade_samples, waveform1.shape[-1], waveform2.shape[-1])
        
        if fade_samples <= 0:
            return torch.cat([waveform1, waveform2], dim=-1)
        
        # Create fade curves
        fade_out = torch.linspace(1.0, 0.0, fade_samples).unsqueeze(0).unsqueeze(0)
        fade_in = torch.linspace(0.0, 1.0, fade_samples).unsqueeze(0).unsqueeze(0)
        
        # Apply fades
        audio1_end = waveform1[..., -fade_samples:] * fade_out
        audio2_start = waveform2[..., :fade_samples] * fade_in
        
        # Create crossfaded section
        crossfaded = audio1_end + audio2_start
        
        # Concatenate: audio1_main + crossfaded + audio2_remaining
        result = torch.cat([
            waveform1[..., :-fade_samples],
            crossfaded,
            waveform2[..., fade_samples:]
        ], dim=-1)
        
        return {
            "waveform": result,
            "sample_rate": sample_rate
        }
    
    def process_audio_queue(self, audio, target_duration, crossfade_duration, queue_id, auto_play):
        import time
        import hashlib
        
        try:
            # Validate and adjust target duration
            validated_duration = self.validate_duration(target_duration)
            
            # Trim audio to precise duration if specified
            if validated_duration > 0:
                processed_audio = self.trim_to_exact_duration(audio, validated_duration)
                actual_duration = validated_duration
            else:
                processed_audio = audio
                actual_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
            
            # Create audio hash for change detection
            audio_hash = hashlib.md5(audio["waveform"].cpu().numpy().tobytes()).hexdigest()
            current_time = time.time()
            
            # Initialize queue state if needed
            if queue_id not in self._audio_buffer:
                self._audio_buffer[queue_id] = []
                self._current_playback[queue_id] = {
                    "end_time": 0,
                    "last_hash": None,
                    "current_audio": None
                }
            
            playback_state = self._current_playback[queue_id]
            buffer = self._audio_buffer[queue_id]
            
            # Check if audio has changed
            audio_changed = playback_state["last_hash"] != audio_hash
            playback_active = current_time < playback_state["end_time"]
            
            if audio_changed and auto_play:
                if not playback_active:
                    # Start playing immediately
                    playback_state["current_audio"] = processed_audio
                    playback_state["end_time"] = current_time + actual_duration
                    playback_state["last_hash"] = audio_hash
                    output_audio = processed_audio
                    status = f"Playing immediately ({actual_duration:.2f}s)"
                else:
                    # Add to buffer for smooth transition
                    buffer.append({
                        "audio": processed_audio,
                        "hash": audio_hash,
                        "duration": actual_duration,
                        "queued_at": current_time
                    })
                    
                    # Create smooth transition if we have current audio
                    if playback_state["current_audio"] is not None:
                        output_audio = self.create_crossfade(
                            playback_state["current_audio"],
                            processed_audio,
                            crossfade_duration
                        )
                        total_duration = (
                            playback_state["current_audio"]["waveform"].shape[-1] + 
                            processed_audio["waveform"].shape[-1] -
                            int(crossfade_duration * processed_audio["sample_rate"])
                        ) / processed_audio["sample_rate"]
                        playback_state["end_time"] = current_time + total_duration
                    else:
                        output_audio = processed_audio
                        playback_state["end_time"] = current_time + actual_duration
                    
                    playback_state["current_audio"] = processed_audio
                    playback_state["last_hash"] = audio_hash
                    status = f"Smooth transition ({crossfade_duration:.1f}s fade)"
            else:
                # Return current audio or silence
                if playback_state["current_audio"] is not None:
                    output_audio = playback_state["current_audio"]
                    remaining_time = max(0, playback_state["end_time"] - current_time)
                    status = f"Current audio ({remaining_time:.1f}s remaining)"
                else:
                    output_audio = processed_audio
                    status = "No active playback"
            
            # Always provide original audio for conditioning
            conditioning_audio = audio
            
            # Create info string
            info = f"Smooth Audio Queue Info:\n"
            info += f"Target duration: {validated_duration:.2f}s (validated)\n"
            info += f"Actual duration: {actual_duration:.2f}s\n"
            info += f"Crossfade: {crossfade_duration:.1f}s\n"
            info += f"Queue ID: {queue_id}\n"
            info += f"Status: {status}\n"
            info += f"Buffered clips: {len(buffer)}\n"
            info += f"Sample rate: {audio['sample_rate']}Hz"
            
            return (output_audio, conditioning_audio, info)
            
        except Exception as e:
            error_msg = f"Error in smooth audio queue: {str(e)}"
            print(error_msg)
            return (audio, audio, error_msg)

class LoadAudioStandalone:
    """Standalone audio loader that doesn't depend on ComfyUI's LoadAudio node"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio_path": ("STRING", {"default": "", "multiline": False})}}
    
    CATEGORY = "audio/musicgen"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    
    def f32_pcm(self, wav):
        """Convert audio to float 32 bits PCM format."""
        if wav.dtype.is_floating_point:
            return wav
        elif wav.dtype == torch.int16:
            return wav.float() / (2 ** 15)
        elif wav.dtype == torch.int32:
            return wav.float() / (2 ** 31)
        raise ValueError(f"Unsupported wav dtype: {wav.dtype}")
    
    def load_audio_file(self, filepath):
        """Load audio file using av library"""
        with av.open(filepath) as af:
            if not af.streams.audio:
                raise ValueError("No audio stream found in the file.")
            
            stream = af.streams.audio[0]
            sr = stream.codec_context.sample_rate
            n_channels = stream.channels
            
            frames = []
            for frame in af.decode(streams=stream.index):
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != n_channels:
                    buf = buf.view(-1, n_channels).t()
                frames.append(buf)
            
            if not frames:
                raise ValueError("No audio frames decoded.")
            
            wav = torch.cat(frames, dim=1)
            wav = self.f32_pcm(wav)
            return wav, sr
    
    def load(self, audio_path):
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        waveform, sample_rate = self.load_audio_file(audio_path)
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


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HuggingFaceMusicGen": HuggingFaceMusicGen,
    "BPMDurationInput": BPMDurationInput,
    "LoopingAudioPreview": LoopingAudioPreview,
    "SmoothAudioQueue": SmoothAudioQueue,
    "MusicGenAudioToFile": MusicGenAudioToFile,
    "SaveAudioStandalone": SaveAudioStandalone,
    "LoadAudioStandalone": LoadAudioStandalone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceMusicGen": "MusicGen (Hugging Face)",
    "BPMDurationInput": "BPM Duration Calculator",
    "LoopingAudioPreview": "Looping Audio Preview",
    "SmoothAudioQueue": "Smooth Audio Queue",
    "MusicGenAudioToFile": "Save MusicGen Audio (Legacy)",
    "SaveAudioStandalone": "Save Audio (WAV/FLAC/MP3/Opus)",
    "LoadAudioStandalone": "Load Audio File"
}
