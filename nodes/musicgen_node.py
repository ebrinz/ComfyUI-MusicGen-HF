"""
MusicGen main generation node
"""

import torch
import numpy as np
import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
try:
    import folder_paths
    import comfy.model_management
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("⚠️ ComfyUI imports not available - using fallback model management")


# Set up ComfyUI model directory integration
if COMFY_AVAILABLE:
    # Create musicgen models directory
    MUSICGEN_MODELS_DIR = os.path.join(folder_paths.models_dir, "musicgen")
    if not os.path.exists(MUSICGEN_MODELS_DIR):
        os.makedirs(MUSICGEN_MODELS_DIR, exist_ok=True)
        print(f"📁 Created MusicGen models directory: {MUSICGEN_MODELS_DIR}")
    
    # Create HuggingFace cache directory within ComfyUI structure
    MUSICGEN_CACHE_DIR = os.path.join(MUSICGEN_MODELS_DIR, "huggingface_cache")
    if not os.path.exists(MUSICGEN_CACHE_DIR):
        os.makedirs(MUSICGEN_CACHE_DIR, exist_ok=True)
    
    # Register with ComfyUI's folder system
    try:
        folder_paths.add_model_folder_path("musicgen", MUSICGEN_MODELS_DIR)
        print(f"✅ Registered MusicGen model path with ComfyUI")
    except:
        # Fallback for older ComfyUI versions
        if "musicgen" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["musicgen"] = ([MUSICGEN_MODELS_DIR], {".safetensors", ".pt", ".pth", ".bin"})
else:
    MUSICGEN_MODELS_DIR = os.path.expanduser("~/.cache/musicgen")
    MUSICGEN_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")


class HuggingFaceMusicGen:
    """
    ComfyUI Node for Facebook's MusicGen via Hugging Face Transformers
    Supports CUDA, MPS (Apple Silicon), and CPU
    Integrates with ComfyUI's model management system
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
        """Determine the best available device using ComfyUI's system if available"""
        if COMFY_AVAILABLE:
            try:
                # Use ComfyUI's device management
                device = comfy.model_management.get_torch_device()
                device_str = str(device).split(':')[0]  # Extract device type (cuda, mps, cpu)
                
                # Verify CUDA is actually available if ComfyUI says to use it
                if device_str == "cuda" and not torch.cuda.is_available():
                    print("⚠️ ComfyUI suggested CUDA but CUDA not available, falling back to CPU")
                    return "cpu"
                    
                return device_str
            except Exception as e:
                print(f"⚠️ ComfyUI device detection failed: {e}")
        
        # Fallback to manual detection with thorough CUDA checking
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                torch.cuda.device_count()
                torch.cuda.get_device_name(0)
                print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
                return "cuda"
            except Exception as e:
                print(f"⚠️ CUDA available but not functional: {e}, falling back")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS functionality
                test_tensor = torch.zeros(1).to("mps")
                del test_tensor
                print("✅ MPS detected and functional")
                return "mps"
            except Exception as e:
                print(f"⚠️ MPS available but not functional: {e}, falling back")
        
        print("✅ Using CPU device")
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
            
            # Use HuggingFace model identifier and let it handle caching automatically
            model_name = f"facebook/musicgen-{model_size}"
            cache_dir = MUSICGEN_CACHE_DIR
            self._loaded_from_local = False
            print(f"🌐 Loading MusicGen model (will use cache if available): {cache_dir}")
            
            try:
                # Load processor (CPU only, no device needed)
                print(f"🔧 Loading processor with PyTorch {torch.__version__}")
                self.processor = AutoProcessor.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    use_safetensors=True  # Force SafeTensors usage
                )
                
                # Get optimal dtype for the device with CUDA capability checking
                if COMFY_AVAILABLE:
                    try:
                        # Use ComfyUI's dtype selection
                        dtype = comfy.model_management.unet_dtype(self.device)
                    except:
                        dtype = torch.float32
                else:
                    # Fallback dtype selection with CUDA capability check
                    if self.device == "cuda":
                        try:
                            # Check if CUDA supports float16
                            if torch.cuda.get_device_capability(0)[0] >= 7:  # Tensor cores for better fp16
                                dtype = torch.float16
                                print("✅ Using float16 with Tensor Core support")
                            else:
                                dtype = torch.float32
                                print("ℹ️ Using float32 (no Tensor Core support)")
                        except:
                            dtype = torch.float32
                            print("ℹ️ Using float32 (CUDA capability check failed)")
                    else:
                        dtype = torch.float32
                
                print(f"💾 Using dtype: {dtype}")
                
                # Load model with unified settings and force SafeTensors
                print(f"🔧 Loading model with dtype {dtype}")
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    use_safetensors=True  # Force SafeTensors usage
                )
                
                # Move model to device with robust error handling
                try:
                    if self.device == "cuda":
                        # Additional CUDA memory checks
                        if hasattr(torch.cuda, 'memory_allocated'):
                            print(f"🔍 CUDA memory before loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
                        
                        # Clear cache before loading large model
                        torch.cuda.empty_cache()
                    
                    self.model = self.model.to(self.device, dtype=None)
                    self.model.eval()  # Set to evaluation mode
                    self.current_model_size = model_size
                    
                    if self.device == "cuda":
                        # Check CUDA memory after loading
                        if hasattr(torch.cuda, 'memory_allocated'):
                            print(f"🔍 CUDA memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
                    
                    # Verify all parameters and buffers are on the correct device
                    device_issues = []
                    for name, buf in self.model.named_buffers():
                        if buf.device.type != self.device:
                            device_issues.append(f"{name}: {buf.device}")
                    
                    if device_issues:
                        print(f"⚠️ Warning: Some buffers not on expected device {self.device}:")
                        for issue in device_issues[:3]:  # Show first 3
                            print(f"   {issue}")
                        if len(device_issues) > 3:
                            print(f"   ... and {len(device_issues) - 3} more")
                    
                    print(f"✅ MusicGen-{model_size} loaded successfully on {self.device}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and self.device == "cuda":
                        print(f"⚠️ CUDA out of memory, falling back to CPU: {e}")
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                        self.model.eval()
                        self.current_model_size = model_size
                        print(f"✅ MusicGen-{model_size} loaded successfully on CPU (fallback)")
                    else:
                        raise e
                
                # Integrate with ComfyUI's memory management
                if COMFY_AVAILABLE:
                    try:
                        # Inform ComfyUI's memory manager about our model
                        comfy.model_management.load_models_gpu([self.model])
                    except:
                        pass
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                
                # Try alternative loading methods
                if "torch.load" in str(e) or "safetensors" in str(e):
                    print("🔄 Trying alternative loading method...")
                    try:
                        # Try without use_safetensors parameter
                        self.processor = AutoProcessor.from_pretrained(
                            model_name, 
                            cache_dir=cache_dir
                        )
                        self.model = MusicgenForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            cache_dir=cache_dir,
                            trust_remote_code=True  # May help with security restrictions
                        )
                        self.model = self.model.to(self.device)
                        print("✅ Alternative loading method succeeded")
                    except Exception as e2:
                        print(f"❌ Alternative method also failed: {e2}")
                        raise e2
                elif self.device != "cpu":
                    # Fallback to CPU if device loading fails
                    print("🔄 Falling back to CPU...")
                    self.device = "cpu"
                    self.model = MusicgenForConditionalGeneration.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float32,
                        cache_dir=cache_dir
                    )
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
                # Audio-prompted generation - handle ComfyUI audio format
                print(f"🎵 Processing conditioning audio...")
                
                # Extract waveform from ComfyUI audio format
                if isinstance(conditioning_audio, dict) and "waveform" in conditioning_audio:
                    conditioning_waveform = conditioning_audio["waveform"]
                    conditioning_sample_rate = conditioning_audio["sample_rate"]
                    print(f"   Conditioning audio shape: {conditioning_waveform.shape}")
                    print(f"   Conditioning sample rate: {conditioning_sample_rate}Hz")
                elif isinstance(conditioning_audio, np.ndarray):
                    conditioning_waveform = torch.from_numpy(conditioning_audio).float()
                    conditioning_sample_rate = 32000  # Default assumption
                    print(f"   Converted numpy array, shape: {conditioning_waveform.shape}")
                else:
                    raise ValueError(f"Unsupported conditioning audio format: {type(conditioning_audio)}")
                
                # Convert to numpy for processor (HuggingFace expects numpy)
                if conditioning_waveform.dim() > 2:
                    # Remove batch dimension if present: (batch, channels, samples) -> (channels, samples)
                    conditioning_waveform = conditioning_waveform.squeeze(0)
                
                if conditioning_waveform.dim() > 1:
                    # Take first channel if stereo: (channels, samples) -> (samples,)
                    conditioning_waveform = conditioning_waveform[0]
                
                # Use the model's expected sample rate
                model_sample_rate = self.model.config.audio_encoder.sampling_rate
                print(f"   Model expects sample rate: {model_sample_rate}Hz")
                
                # Resample if necessary
                if conditioning_sample_rate != model_sample_rate:
                    print(f"   Resampling from {conditioning_sample_rate}Hz to {model_sample_rate}Hz...")
                    import torchaudio.functional as F
                    conditioning_waveform = F.resample(
                        conditioning_waveform, 
                        conditioning_sample_rate, 
                        model_sample_rate
                    )
                    print(f"   Resampled shape: {conditioning_waveform.shape}")
                
                # Convert to numpy array for the processor
                conditioning_numpy = conditioning_waveform.cpu().numpy().astype(np.float32)
                print(f"   Final conditioning shape for processor: {conditioning_numpy.shape}")
                
                inputs = self.processor(
                    audio=conditioning_numpy,
                    sampling_rate=model_sample_rate,
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
            try:
                inputs = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                          for k, v in inputs.items()}
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    print(f"⚠️ CUDA out of memory during input processing, clearing cache...")
                    torch.cuda.empty_cache()
                    # Try again with blocking transfer
                    inputs = {k: v.to(self.device, non_blocking=False) if torch.is_tensor(v) else v
                              for k, v in inputs.items()}
                else:
                    raise e
            
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
            
            # Store generated audio for conditioning queue system
            try:
                from .conditioning_queue_nodes import AudioOutputToConditioningQueue
                AudioOutputToConditioningQueue.store_generated_audio("main_chain", audio_output)
            except ImportError:
                pass  # Conditioning queue node not available
            
            # Create info string
            actual_duration = len(audio_data) / sampling_rate
            duration_source = "BPM input" if duration_override > 0.0 else "manual"
            conditioning_info = "None"
            if conditioning_audio is not None:
                conditioning_info = "✅ Used conditioning audio"
            
            # Model path info
            model_location = "HuggingFace cache (ComfyUI managed)"
            
            info = f"Generated {actual_duration:.1f}s audio using MusicGen-{model_size} on {self.device}\n"
            info += f"Model location: {model_location}\n"
            info += f"Duration source: {duration_source} ({effective_duration:.2f}s requested)\n"
            info += f"Conditioning: {conditioning_info}\n"
            info += f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
            info += f"Settings: guidance={guidance_scale}, tokens={tokens_to_use}, seed={seed}\n"
            info += f"Sample rate: {sampling_rate}Hz"
            
            return (audio_output, info)
            
        except Exception as e:
            error_msg = f"Error generating audio: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Provide more specific error information for conditioning audio issues
            if conditioning_audio is not None:
                print(f"🔍 Conditioning audio debug info:")
                print(f"   Type: {type(conditioning_audio)}")
                if isinstance(conditioning_audio, dict):
                    print(f"   Keys: {list(conditioning_audio.keys())}")
                    if "waveform" in conditioning_audio:
                        print(f"   Waveform shape: {conditioning_audio['waveform'].shape}")
                        print(f"   Waveform dtype: {conditioning_audio['waveform'].dtype}")
                    if "sample_rate" in conditioning_audio:
                        print(f"   Sample rate: {conditioning_audio['sample_rate']}")
                elif hasattr(conditioning_audio, 'shape'):
                    print(f"   Shape: {conditioning_audio.shape}")
                    print(f"   Dtype: {conditioning_audio.dtype}")
            
            # Return silence on error in ComfyUI AUDIO format
            empty_audio = torch.zeros([1, 1, int(32000 * effective_duration)], dtype=torch.float32)  # 32kHz silence
            empty_audio_output = {
                "waveform": empty_audio,
                "sample_rate": 32000
            }
            return (empty_audio_output, error_msg)