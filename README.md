# ComfyUI MusicGen (Hugging Face)

A **standalone** ComfyUI custom node package for Facebook's MusicGen using Hugging Face Transformers. Generate high-quality music from text prompts with full support for CUDA, MPS (Apple Silicon), and CPU.

## ✨ Features

- 🎵 **Text-to-Music Generation** - Create music from natural language descriptions
- 🎛️ **Multiple Model Sizes** - Choose between small, medium, and large models
- 🍎 **Apple Silicon Support** - Full MPS acceleration for M1/M2/M3 Macs
- 🚀 **Multi-Device Support** - Automatic device detection (CUDA/MPS/CPU)
- 🎚️ **Advanced Controls** - Guidance scale, sampling, temperature controls
- 📁 **Multi-Format Export** - Save as WAV, FLAC, MP3, or Opus with quality settings
- 🔧 **Standalone Operation** - No dependencies on ComfyUI core modifications
- 🎧 **Built-in Audio Utilities** - Complete audio loading and saving capabilities

## 🚀 Installation

### Option 1: Clone from GitHub
```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/your-username/ComfyUI-MusicGen-HF.git
cd ComfyUI-MusicGen-HF
pip install -r requirements.txt
```

### Option 2: Manual Installation
1. **Navigate to your ComfyUI custom nodes directory:**
   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   ```

2. **Create the directory:**
   ```bash
   mkdir ComfyUI-MusicGen-HF
   cd ComfyUI-MusicGen-HF
   ```

3. **Install dependencies:**
   ```bash
   pip install transformers>=4.30.0 accelerate>=0.20.0 scipy>=1.10.0 torch>=2.0.0 torchaudio>=2.0.0 av>=15.0.0
   ```

4. **Add the node files** (copy `nodes.py`, `__init__.py`, and `requirements.txt`)

5. **Restart ComfyUI**

> **Note**: This is a completely standalone package - no modifications to ComfyUI core files are required!

## 🎵 Usage

### Available Nodes

- **MusicGen (Hugging Face)** - Main text-to-music generation node
- **Save Audio (WAV/FLAC/MP3/Opus)** - Advanced audio saver with format options
- **Load Audio File** - Load audio files for conditioning or processing
- **Save MusicGen Audio (Legacy)** - Simplified audio saver for basic use

### Basic Workflow

1. **Add MusicGen Node:**
   - In ComfyUI, add `MusicGen (Hugging Face)` node
   - Enter your text prompt (e.g., "upbeat electronic music with drums")
   - Choose model size (start with "small" for faster generation)
   - Set duration (1-30 seconds)

2. **Add Save Node:**
   - Add `Save Audio (WAV/FLAC/MP3/Opus)` node for full format control
   - OR add `Save MusicGen Audio (Legacy)` for simple saving
   - Connect the audio output from MusicGen
   - Choose format (WAV, FLAC, MP3, Opus) and quality settings

3. **Generate:**
   - Click "Queue Prompt"
   - First run will download the model (~1.5GB for small)
   - Generated audio saves to `ComfyUI/output/`

### Audio Conditioning (Optional)

1. **Load Reference Audio:**
   - Add `Load Audio File` node
   - Specify path to reference audio file
   - Connect to MusicGen's `conditioning_audio` input

2. **Generate Continuation:**
   - The model will generate music that continues/extends the reference audio

### Model Sizes

- **Small** (~1.5GB): Fast generation, good quality
- **Medium** (~3.3GB): Better quality, slower generation  
- **Large** (~3.3GB): Best quality, slowest generation

### Device Support

The node automatically detects and uses the best available device:

- **CUDA** (NVIDIA GPUs): Full acceleration with mixed precision
- **MPS** (Apple Silicon): Optimized for M1/M2/M3 chips
- **CPU**: Fallback for compatibility

### Audio Format Options

| Format | Quality Settings | Best For |
|--------|------------------|----------|
| **WAV** | Uncompressed | Highest quality, large files |
| **FLAC** | Lossless | High quality, smaller than WAV |
| **MP3** | 128k, 192k, 320k, V0 | Good compression, wide compatibility |
| **Opus** | 64k, 96k, 128k, 192k, 320k | Best compression, modern standard |

### Parameters

- **prompt**: Text description of the music to generate
- **model_size**: small/medium/large model variants
- **duration**: Length of generated audio (1-30 seconds)
- **guidance_scale**: Higher values follow prompt more closely (1-10)
- **do_sample**: Enable sampling for more diverse outputs
- **max_new_tokens**: Maximum tokens to generate (50-1503)
- **seed**: Random seed for reproducible generation
- **temperature**: Sampling randomness (0.1-2.0)
- **conditioning_audio**: Optional reference audio for continuation

## Example Prompts

- "80s pop track with bassy drums and synth"
- "classical piano piece in minor key"
- "upbeat jazz with saxophone solo"
- "ambient electronic soundscape"
- "acoustic guitar fingerpicking folk song"

## Troubleshooting

### MPS Issues
If you encounter MPS errors:
- The node will automatically fallback to CPU
- Ensure macOS 12.3+ and PyTorch 2.0+

### Memory Issues
- Start with "small" model
- Reduce `max_new_tokens` 
- Lower `duration`

### Model Download
- First run downloads models to `~/.cache/huggingface/`
- Ensure stable internet connection
- Models are cached for subsequent use

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 2-4GB for model cache
- **GPU**: Optional but recommended (CUDA/MPS support)

### Dependencies
All dependencies are automatically installed via `requirements.txt`:
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing
- `transformers>=4.30.0` - Hugging Face models
- `accelerate>=0.20.0` - Model optimization
- `scipy>=1.10.0` - Scientific computing
- `av>=15.0.0` - Audio/video processing

### Compatibility
- ✅ **Windows** (CUDA/CPU)
- ✅ **macOS** (MPS/CPU) - Intel and Apple Silicon
- ✅ **Linux** (CUDA/CPU)
- ✅ **ComfyUI** - All recent versions

## 🔗 Links

- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [Hugging Face Model Hub](https://huggingface.co/facebook/musicgen-small)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---
