"""
Advanced loop and transition nodes for professional dance music workflows
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import hashlib
import math


class ProfessionalLoopTransition:
    """
    Professional-grade loop transition node designed for dance music
    Features: DC offset removal, beat-aware crossfading, phase alignment, zero-crossing detection
    """
    
    # Class-level state for seamless playback
    _playback_state = {}
    _loop_buffer = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "bpm": ("FLOAT", {
                    "default": 128.0,
                    "min": 60.0,
                    "max": 200.0,
                    "step": 0.1,
                    "tooltip": "BPM for beat-aligned transitions"
                }),
                "transition_type": (["beat_match", "phrase_match", "zero_cross", "power_match"], {
                    "default": "beat_match",
                    "tooltip": "Type of transition algorithm"
                }),
                "transition_beats": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 8.0,
                    "step": 0.25,
                    "tooltip": "Transition duration in beats"
                }),
                "loop_id": ("STRING", {
                    "default": "main_loop",
                    "tooltip": "Loop identifier for multiple streams"
                }),
                "auto_normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically normalize audio levels"
                }),
                "dc_offset_removal": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove DC offset before transitions"
                })
            },
            "optional": {
                "crossfade_curve": (["linear", "equal_power", "exponential", "s_curve"], {
                    "default": "equal_power",
                    "tooltip": "Crossfade curve type"
                }),
                "phase_align": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Align phases for smoother transitions"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("looped_audio", "conditioning_audio", "info")
    FUNCTION = "process_professional_loop"
    CATEGORY = "audio/musicgen"
    OUTPUT_NODE = True
    
    def remove_dc_offset(self, waveform):
        """Remove DC offset from audio waveform"""
        if waveform.numel() == 0:
            return waveform
        
        # Calculate DC offset (mean)
        dc_offset = torch.mean(waveform, dim=-1, keepdim=True)
        
        # Remove DC offset
        return waveform - dc_offset
    
    def find_zero_crossings(self, waveform, window_samples=1024):
        """Find zero crossings near specified positions for smooth cuts"""
        if waveform.numel() < window_samples:
            return 0
        
        # Look for zero crossings in a window
        start_idx = max(0, len(waveform) - window_samples)
        end_idx = len(waveform)
        
        window = waveform[start_idx:end_idx]
        
        # Find sign changes (zero crossings)
        signs = torch.sign(window)
        zero_crossings = torch.where(signs[:-1] != signs[1:])[0]
        
        if len(zero_crossings) > 0:
            # Return the last zero crossing position (closest to end)
            return start_idx + zero_crossings[-1].item()
        
        return len(waveform) - 1
    
    def calculate_beat_samples(self, bpm, sample_rate, beats):
        """Calculate number of samples for given beats at BPM"""
        seconds_per_beat = 60.0 / bpm
        beat_duration_seconds = beats * seconds_per_beat
        return int(beat_duration_seconds * sample_rate)
    
    def create_crossfade_curve(self, length, curve_type="equal_power"):
        """Create different types of crossfade curves"""
        t = torch.linspace(0, 1, length)
        
        if curve_type == "linear":
            fade_out = 1 - t
            fade_in = t
        elif curve_type == "equal_power":
            # Equal power crossfade - maintains constant power
            fade_out = torch.cos(t * math.pi / 2)
            fade_in = torch.sin(t * math.pi / 2)
        elif curve_type == "exponential":
            fade_out = torch.pow(1 - t, 2)
            fade_in = torch.pow(t, 2)
        elif curve_type == "s_curve":
            # S-curve for smooth transitions
            fade_out = 0.5 * (1 + torch.cos(t * math.pi))
            fade_in = 0.5 * (1 - torch.cos(t * math.pi))
        else:
            # Default to equal power
            fade_out = torch.cos(t * math.pi / 2)
            fade_in = torch.sin(t * math.pi / 2)
        
        return fade_out.unsqueeze(0).unsqueeze(0), fade_in.unsqueeze(0).unsqueeze(0)
    
    def align_phases(self, waveform1, waveform2, max_shift=1024):
        """Align phases between two waveforms for smoother transitions"""
        if waveform1.shape[-1] < max_shift or waveform2.shape[-1] < max_shift:
            return waveform1, waveform2
        
        # Take a window from the end of waveform1 and beginning of waveform2
        w1_end = waveform1[..., -max_shift:]
        w2_start = waveform2[..., :max_shift]
        
        # Flatten for correlation calculation
        w1_flat = w1_end.flatten()
        w2_flat = w2_start.flatten()
        
        if w1_flat.numel() == 0 or w2_flat.numel() == 0:
            return waveform1, waveform2
        
        # Find best alignment using cross-correlation
        correlation = F.conv1d(
            w2_flat.unsqueeze(0).unsqueeze(0),
            w1_flat.flip(0).unsqueeze(0).unsqueeze(0),
            padding=max_shift//2
        )
        
        best_shift = torch.argmax(correlation) - max_shift//2
        best_shift = best_shift.item()
        
        # Apply shift to waveform2
        if best_shift > 0:
            # Shift right (delay)
            shifted_w2 = F.pad(waveform2, (best_shift, 0))[..., :-best_shift]
        elif best_shift < 0:
            # Shift left (advance)
            shifted_w2 = F.pad(waveform2, (0, -best_shift))[..., -best_shift:]
        else:
            shifted_w2 = waveform2
        
        return waveform1, shifted_w2
    
    def create_professional_transition(self, audio1, audio2, bpm, transition_beats, 
                                     transition_type, crossfade_curve, phase_align, sample_rate):
        """Create professional-grade transition between two audio clips"""
        
        waveform1 = audio1["waveform"]
        waveform2 = audio2["waveform"]
        
        # Remove DC offset
        waveform1 = self.remove_dc_offset(waveform1)
        waveform2 = self.remove_dc_offset(waveform2)
        
        # Calculate transition duration in samples
        transition_samples = self.calculate_beat_samples(bpm, sample_rate, transition_beats)
        
        # Ensure we don't exceed audio lengths
        max_transition = min(waveform1.shape[-1], waveform2.shape[-1], transition_samples)
        
        if max_transition <= 0:
            return torch.cat([waveform1, waveform2], dim=-1)
        
        # Apply phase alignment if requested
        if phase_align:
            waveform1, waveform2 = self.align_phases(waveform1, waveform2, max_transition)
        
        if transition_type == "zero_cross":
            # Find optimal zero crossing points
            cut_point1 = self.find_zero_crossings(waveform1[0, 0], max_transition)
            cut_point2 = self.find_zero_crossings(waveform2[0, 0], max_transition)
            
            # Adjust transition samples based on zero crossings
            transition_samples = min(waveform1.shape[-1] - cut_point1, cut_point2, max_transition)
            
            if transition_samples > 0:
                # Create crossfade curves
                fade_out, fade_in = self.create_crossfade_curve(transition_samples, crossfade_curve)
                
                # Apply crossfade
                audio1_end = waveform1[..., cut_point1:cut_point1 + transition_samples] * fade_out
                audio2_start = waveform2[..., :transition_samples] * fade_in
                
                crossfaded = audio1_end + audio2_start
                
                # Concatenate with zero-crossing alignment
                result = torch.cat([
                    waveform1[..., :cut_point1],
                    crossfaded,
                    waveform2[..., transition_samples:]
                ], dim=-1)
            else:
                result = torch.cat([waveform1, waveform2], dim=-1)
                
        elif transition_type == "power_match":
            # Match power levels before crossfading
            power1 = torch.mean(waveform1[..., -max_transition:] ** 2)
            power2 = torch.mean(waveform2[..., :max_transition] ** 2)
            
            if power2 > 0:
                power_ratio = torch.sqrt(power1 / power2)
                waveform2 = waveform2 * power_ratio
            
            # Standard crossfade with power matching
            fade_out, fade_in = self.create_crossfade_curve(max_transition, crossfade_curve)
            
            audio1_end = waveform1[..., -max_transition:] * fade_out
            audio2_start = waveform2[..., :max_transition] * fade_in
            
            crossfaded = audio1_end + audio2_start
            
            result = torch.cat([
                waveform1[..., :-max_transition],
                crossfaded,
                waveform2[..., max_transition:]
            ], dim=-1)
            
        else:  # beat_match or phrase_match
            # Standard beat-aligned crossfade
            fade_out, fade_in = self.create_crossfade_curve(max_transition, crossfade_curve)
            
            audio1_end = waveform1[..., -max_transition:] * fade_out
            audio2_start = waveform2[..., :max_transition] * fade_in
            
            crossfaded = audio1_end + audio2_start
            
            result = torch.cat([
                waveform1[..., :-max_transition],
                crossfaded,
                waveform2[..., max_transition:]
            ], dim=-1)
        
        return {
            "waveform": result,
            "sample_rate": sample_rate
        }
    
    def process_professional_loop(self, audio, bpm, transition_type, transition_beats, loop_id, 
                                auto_normalize, dc_offset_removal, crossfade_curve="equal_power", phase_align=True):
        
        try:
            # Extract audio properties
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Apply DC offset removal if requested
            if dc_offset_removal:
                waveform = self.remove_dc_offset(waveform)
            
            # Apply normalization if requested
            if auto_normalize:
                max_val = torch.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val * 0.95
            
            # Update audio dict
            processed_audio = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            # Audio change detection
            audio_hash = hashlib.md5(waveform.cpu().numpy().tobytes()).hexdigest()
            current_time = time.time()
            
            # Initialize loop state
            if loop_id not in self._playback_state:
                self._playback_state[loop_id] = {
                    "last_hash": None,
                    "current_audio": None,
                    "transition_active": False
                }
                self._loop_buffer[loop_id] = []
            
            state = self._playback_state[loop_id]
            
            # Check if audio has changed
            audio_changed = state["last_hash"] != audio_hash
            
            if audio_changed:
                if state["current_audio"] is not None:
                    # Create professional transition
                    output_audio = self.create_professional_transition(
                        state["current_audio"],
                        processed_audio,
                        bpm,
                        transition_beats,
                        transition_type,
                        crossfade_curve,
                        phase_align,
                        sample_rate
                    )
                    status = f"Professional transition: {transition_type} ({transition_beats} beats)"
                else:
                    output_audio = processed_audio
                    status = "First loop - no transition"
                
                # Update state
                state["current_audio"] = processed_audio
                state["last_hash"] = audio_hash
                
            else:
                # Same audio, return current
                output_audio = state["current_audio"] if state["current_audio"] else processed_audio
                status = "Loop maintained"
            
            # Always provide original audio for conditioning
            conditioning_audio = audio
            
            # Calculate transition duration
            transition_duration = (transition_beats / bpm) * 60.0
            audio_duration = waveform.shape[-1] / sample_rate
            
            # Create info string
            info = f"Professional Loop Transition:\n"
            info += f"BPM: {bpm}, Transition: {transition_beats} beats ({transition_duration:.2f}s)\n"
            info += f"Type: {transition_type}, Curve: {crossfade_curve}\n"
            info += f"DC Offset Removal: {dc_offset_removal}, Phase Align: {phase_align}\n"
            info += f"Audio Duration: {audio_duration:.2f}s\n"
            info += f"Status: {status}\n"
            info += f"Loop ID: {loop_id}"
            
            return (output_audio, conditioning_audio, info)
            
        except Exception as e:
            error_msg = f"Error in professional loop transition: {str(e)}"
            print(f"❌ {error_msg}")
            return (audio, audio, error_msg)