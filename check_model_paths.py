#!/usr/bin/env python3
"""
Script to check MusicGen model storage paths
"""

import os
import sys

def check_model_paths():
    print("🔍 MusicGen Model Storage Path Check")
    print("=" * 50)
    
    # Try to import ComfyUI folder_paths
    try:
        sys.path.insert(0, "/Users/crashy/Repositories/ComfyUI")
        import folder_paths
        print("✅ ComfyUI found - using integrated model management")
        
        # Show ComfyUI models directory
        models_dir = folder_paths.models_dir
        print(f"📁 ComfyUI models directory: {models_dir}")
        
        # Show MusicGen directory
        musicgen_dir = os.path.join(models_dir, "musicgen")
        print(f"🎵 MusicGen models directory: {musicgen_dir}")
        print(f"   Exists: {'✅' if os.path.exists(musicgen_dir) else '❌'}")
        
        # Show HuggingFace cache within ComfyUI
        cache_dir = os.path.join(musicgen_dir, "huggingface_cache")
        print(f"🌐 HuggingFace cache directory: {cache_dir}")
        print(f"   Exists: {'✅' if os.path.exists(cache_dir) else '❌'}")
        
        return musicgen_dir, cache_dir
        
    except ImportError:
        print("⚠️ ComfyUI not found - using fallback paths")
        
        # Fallback directories
        musicgen_dir = os.path.expanduser("~/.cache/musicgen")
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        
        print(f"🎵 MusicGen models directory: {musicgen_dir}")
        print(f"🌐 HuggingFace cache directory: {cache_dir}")
        
        return musicgen_dir, cache_dir

def show_current_models():
    print("\n🔍 Current Model Status")
    print("=" * 50)
    
    # Check HuggingFace default cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache):
        print(f"📦 HuggingFace default cache: {hf_cache}")
        models = [d for d in os.listdir(hf_cache) if "musicgen" in d.lower()]
        if models:
            print("🎵 Found MusicGen models:")
            for model in models:
                print(f"   - {model}")
        else:
            print("❌ No MusicGen models found in default cache")
    else:
        print("❌ No HuggingFace cache found")

def show_recommendations():
    print("\n💡 Model Management Recommendations")
    print("=" * 50)
    print("1. 🎯 Preferred: Place models in ComfyUI/models/musicgen/")
    print("   - Easy user access and management")
    print("   - Integrated with ComfyUI's system")
    print("   - Consistent with other custom nodes")
    print()
    print("2. 🔄 Automatic: Let the node download to huggingface_cache/")
    print("   - First run will download automatically")
    print("   - Stored within ComfyUI structure")
    print("   - Cached for future use")
    print()
    print("3. 📁 Manual: Copy existing models from ~/.cache/huggingface/")
    print("   - Avoid re-downloading if you have models")
    print("   - Move to ComfyUI/models/musicgen/ for better organization")

if __name__ == "__main__":
    musicgen_dir, cache_dir = check_model_paths()
    show_current_models()
    show_recommendations()
    
    print(f"\n✨ Models will be stored in: {musicgen_dir}")
    print(f"🌐 Downloads will cache to: {cache_dir}")