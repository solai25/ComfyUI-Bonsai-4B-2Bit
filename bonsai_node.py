import os
import sys
import io
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
import folder_paths

# --- 1. System Path Setup ---
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(NODE_DIR)

try:
    from backend_gpu.pipeline_gpu import GpuPipeline
except ImportError:
    raise ImportError("❌ Could not find backend_gpu. Make sure it is copied into the ComfyUI-Bonsai-4B folder.")

# --- 2. Global Pipeline State ---
_GLOBAL_BONSAI_PIPE = None

class BonsaiTernaryNode:
    DESCRIPTION = "Bonsai 4B (1-Bit/2-Bit) Image Generation Node. Created by Solai25."
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Dropdown menu to choose between 2-Bit and 1-Bit variants
                "model_type": (["Bonsai-4B (2-Bit Ternary)", "Bonsai-4B (1-Bit Binary)"], {"default": "Bonsai-4B (2-Bit Ternary)"}),
                "prompt": ("STRING", {"multiline": True, "default": "A high-resolution photograph of a bonsai tree in an ancient temple..."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                # Kept as string input, pointing to your base ComfyUI models directory
                "model_folder_path": ("STRING", {"default": folder_paths.models_dir}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "🌳 Bonsai Image 4B"

    def load_pipeline(self, base_models_dir, target_backend):
        global _GLOBAL_BONSAI_PIPE
        
        # 1. Establish path mappings pointing to the SINGLE unified Bonsai-4B folder
        bonsai_4b_dir = os.path.join(base_models_dir, "Bonsai-4B")

        ternary_path = os.path.join(bonsai_4b_dir, "transformer-gemlite-int2")
        binary_path = os.path.join(bonsai_4b_dir, "transformer-gemlite-int1") # Now lives alongside int2
        text_enc_path = os.path.join(bonsai_4b_dir, "text_encoder-hqq-4bit")
        vae_path = os.path.join(bonsai_4b_dir, "vae")
        tokenizer_path = os.path.join(bonsai_4b_dir, "text_encoder-hqq-4bit", "tokenizer")

        # 2. Auto-Download for the 2-Bit baseline if Text Encoder or VAE are missing
        if not os.path.exists(text_enc_path) or not os.path.exists(vae_path) or (target_backend == "bonsai-ternary-gemlite" and not os.path.exists(ternary_path)):
            print(f"🌍 [Bonsai Node] Core components or 2-Bit weights missing at {bonsai_4b_dir}. Downloading from Hugging Face...")
            try:
                snapshot_download(
                    repo_id="prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
                    local_dir=bonsai_4b_dir,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", "*.git*"]
                )
                print("✅ [Bonsai Node] Shared assets & 2-Bit weights downloaded.")
            except Exception as e:
                print(f"❌ [Bonsai Node] 2-Bit core download failed: {e}")
                raise e

        # 3. Surgical Auto-Download ONLY for the 1-Bit Transformer
        if target_backend == "bonsai-binary-gemlite" and not os.path.exists(binary_path):
            print(f"🌍 [Bonsai Node] 1-Bit Transformer missing. Downloading ONLY the 1-bit weights to {binary_path}...")
            try:
                snapshot_download(
                    repo_id="prism-ml/bonsai-image-binary-4B-gemlite-1bit",
                    local_dir=bonsai_4b_dir, # Target the main shared directory
                    allow_patterns=["transformer-gemlite-int1/*"], # Magic filter: ONLY grab this folder
                    local_dir_use_symlinks=False
                )
                print("✅ [Bonsai Node] 1-Bit Transformer downloaded successfully.")
            except Exception as e:
                print(f"❌ [Bonsai Node] 1-Bit Transformer download failed: {e}")
                raise e

        # 4. Deep structural fallback scan inside the unified directory
        for root, dirs, files in os.walk(bonsai_4b_dir):
            if "gemlite_autotune.json" in files and "int2" in root.lower(): ternary_path = root
            if "gemlite_autotune.json" in files and "int1" in root.lower(): binary_path = root
            if "config.json" in files and "text_encoder" in root.lower(): text_enc_path = root
            if "config.json" in files and "vae" in root.lower(): vae_path = root
            if "tokenizer_config.json" in files: tokenizer_path = root

        # 5. Initialize or dynamically hot-swap the pipeline environment
        if _GLOBAL_BONSAI_PIPE is None or getattr(_GLOBAL_BONSAI_PIPE, "ready", False) == False:
            print(f"🌳 [Bonsai Node] Initializing GpuPipeline with backend: {target_backend}")
            
            temp_pipe = GpuPipeline(
                backend=target_backend,
                ternary_transformer_path=ternary_path,
                binary_transformer_path=binary_path, 
                text_encoder_path=text_enc_path,
                vae_path=vae_path,
                tokenizer_path=tokenizer_path,
                device="cuda:0"
            )
            temp_pipe.prewarm()
            _GLOBAL_BONSAI_PIPE = temp_pipe
            print("✅ [Bonsai Node] GPU pipeline loaded and warm.")
        else:
            # Hot-swapping the active transformer on the fly if the user changes the dropdown option
            chosen_transformer = binary_path if target_backend == "bonsai-binary-gemlite" else ternary_path
            _GLOBAL_BONSAI_PIPE.ensure_backend(backend=target_backend, model_path=str(chosen_transformer))
            print(f"✅ [Bonsai Node] Successfully swapped to {target_backend}.")
            
        return _GLOBAL_BONSAI_PIPE

    def generate(self, model_type, prompt, width, height, steps, guidance, seed, model_folder_path):
        # Map human-readable dropdown choices to native pipeline backends
        target_backend = "bonsai-binary-gemlite" if "1-Bit" in model_type else "bonsai-ternary-gemlite"
        
        # Resolve assets and ensure backend alignment
        pipe = self.load_pipeline(model_folder_path, target_backend)
        
        print(f"⚡ [Bonsai Node] Operating on {target_backend} | Running seed: {seed}")
        
        png_bytes = pipe.generate_png(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed
        )
        
        pil_image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]
        
        return (image_tensor,)
