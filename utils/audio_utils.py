"""
Shared audio processing utilities for MusicGen nodes
"""

import torch
import torchaudio
import numpy as np
import tempfile
import os
import folder_paths
import av
import io


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


def f32_pcm(wav):
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load_audio_file(filepath):
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
        wav = f32_pcm(wav)
        return wav, sr