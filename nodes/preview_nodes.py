"""
Audio preview and queue management nodes
"""

import torch
import time
import hashlib


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