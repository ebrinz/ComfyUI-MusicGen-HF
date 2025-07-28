"""
BPM and musical timing nodes
"""


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