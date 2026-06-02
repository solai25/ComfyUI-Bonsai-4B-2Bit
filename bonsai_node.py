from huggingface_hub import snapshot_download
import os
import sys
import io
import torch
import numpy as np
from PIL import Image
import folder_paths

# --- 1. System Path Setup ---
# Force Python to recognize the local backend_gpu folder
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(NODE_DIR)

try:
    from backend_gpu.pipeline_gpu import GpuPipeline
except ImportError:
    raise ImportError("❌ Could not find backend_gpu. Make sure it is copied into the ComfyUI-Bonsai-4B folder.")

# --- 2. Global Pipeline State ---
# Kept global to prevent reloading the model on every prompt execution
_GLOBAL_BONSAI_PIPE = None

class BonsaiTernaryNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        # Dynamically determine the native ComfyUI\models\Bonsai-4B path
        native_bonsai_path = os.path.join(folder_paths.models_dir, "Bonsai-4B")
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A high-resolution photograph of a bonsai tree in an ancient temple..."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                # Automatically defaults to your native ComfyUI model folder
                "model_folder_path": ("STRING", {"default": native_bonsai_path}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Bonsai Image"

    def load_pipeline(self, base_model_dir):
        global _GLOBAL_BONSAI_PIPE

        # 1. Check if the directory exists, if not, download it automatically
        if not os.path.exists(base_model_dir):
            print(f"🌍 [Bonsai Node] Model not found at {base_model_dir}. Starting download...")
            try:
                # This downloads the specific repository to your ComfyUI models folder
                snapshot_download(
                    repo_id="prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
                    local_dir=base_model_dir,
                    local_dir_use_symlinks=False
                )
                print("✅ [Bonsai Node] Download complete!")
            except Exception as e:
                print(f"❌ [Bonsai Node] Download failed: {e}")
                raise e
            
        # 2. Check if the pipeline exists AND is actually fully pre-warmed
        if _GLOBAL_BONSAI_PIPE is not None and getattr(_GLOBAL_BONSAI_PIPE, "ready", False):
            return _GLOBAL_BONSAI_PIPE
            
        print(f"🌳 [Bonsai Node] Initializing Gemlite/HQQ Kernels from: {base_model_dir}")
        
        # Map subfolders inside ComfyUI\models\Bonsai-4B
        ternary_path = os.path.join(base_model_dir, "transformer-gemlite-int2")
        text_enc_path = os.path.join(base_model_dir, "text_encoder-hqq-4bit")
        vae_path = os.path.join(base_model_dir, "vae")
        tokenizer_path = os.path.join(base_model_dir, "text_encoder-hqq-4bit", "tokenizer")

        # Fallback scanners in case the folders are structured differently inside the directory
        if not os.path.exists(ternary_path):
            for root, dirs, files in os.walk(base_model_dir):
                if "gemlite_autotune.json" in files: ternary_path = root
                if "config.json" in files and "text_encoder" in root.lower(): text_enc_path = root
                if "config.json" in files and "vae" in root.lower(): vae_path = root
                if "tokenizer_config.json" in files: tokenizer_path = root

        # 3. Use a temporary variable so a crash doesn't corrupt the global state
        temp_pipe = GpuPipeline(
            backend="bonsai-ternary-gemlite",
            ternary_transformer_path=ternary_path,
            binary_transformer_path=ternary_path, 
            text_encoder_path=text_enc_path,
            vae_path=vae_path,
            tokenizer_path=tokenizer_path,
            device="cuda:0"
        )
        
        # Execute the warmup phase
        temp_pipe.prewarm()
        
        # 4. Only save it to the global cache AFTER a successful prewarm
        _GLOBAL_BONSAI_PIPE = temp_pipe
        print("✅ [Bonsai Node] Loaded onto GPU successfully.")
        
        return _GLOBAL_BONSAI_PIPE

    def generate(self, prompt, width, height, steps, guidance, seed, model_folder_path):
        # Ensure the paths are loaded dynamically
        pipe = self.load_pipeline(model_folder_path)
        
        print(f"⚡ [Bonsai Node] Generating image for seed: {seed}")
        
        # Call the generation engine
        png_bytes = pipe.generate_png(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed
        )
        
        # Convert raw output PNG bytes back to an uncompressed PIL frame
        pil_image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        
        # Parse PIL into an internal ComfyUI PyTorch Tensor arrangement [Batch, Height, Width, Channel]
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]
        
        return (image_tensor,)
