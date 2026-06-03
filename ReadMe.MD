ComfyUI-Bonsai-4B (1-Bit & 2-Bit)

<img height="350" alt="Custom Resolution 🐲" src="example_workflow\Bonsai-image-4B.png" />

A custom ComfyUI node to run the Bonsai 4B (1-bit Binary & 2-bit Ternary) image generation models locally. This node utilizes optimized Gemlite and HQQ kernels for high-speed inference on Windows/Linux GPU environments, allowing you to hot-swap between models instantly without reloading shared assets.

⚠️ Prerequisites

This node requires specific 2-bit/4-bit quantization libraries. Before installing, ensure you have the following in your ComfyUI environment:


```Bash
pip install hqq gemlite triton-windows
```

## 🛠️ Installation

1. Navigate to your ComfyUI/custom_nodes/ directory.

2. Clone this repository:
    
    ```
    git clone https://github.com/solai25/ComfyUI-Bonsai-4B-2Bit
    ```

3. Install Dependencies:

   * For Standard Python (venv/Global):
   
   ```Bash
    cd ComfyUI-Bonsai-4B
    pip install -r requirements.txt
    ```

   * For ComfyUI Portable (Windows):
   
   If you are using the portable version, open your ComfyUI_windows_portable folder and run the command using the internal Python executable:
    
    
    ```Bash
    .\python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-Bonsai-4B\requirements.txt
    ```

###(Optional) If you don't use Git, download the folder and place it in your custom_nodes/ directory.

## 📥 Model Setup

The node expects the model files to be organized in your ComfyUI directory:

```
    ComfyUI/
    └── models/
        └── Bonsai-4B/   <-- Create this folder
            ├── text_encoder-hqq-4bit/       (Shared)
            ├── vae/                         (Shared)
            ├── transformer-gemlite-int2/    (2-Bit Weights)
            └── transformer-gemlite-int1/    (1-Bit Weights)
```


Note: If any component is missing, the node features surgical auto-downloading. If you select a model type (e.g., 1-Bit) that isn't downloaded yet, it will automatically fetch only the specific transformer folder from Hugging Face into your Bonsai-4B directory, saving disk space by reusing the Text Encoder and VAE.

## 🚀 Usage

* Restart ComfyUI.
* Right-click the canvas and select Add Node > Bonsai Image > Bonsai 4B.
* Use the model_type dropdown to switch seamlessly between Bonsai-4B (2-Bit Ternary) and Bonsai-4B (1-Bit Binary).
* Connect the IMAGE output to a Save Image node.
* Configure your prompt and inference settings.
* Queue Prompt. The model will compile and warm up on the GPU for the first generation. Subsequent swaps between 1-Bit and 2-Bit are instant.

## 📝 Troubleshooting

* **Missing Kernels:** Ensure triton is installed correctly for your specific GPU architecture.

* **RuntimeError (prewarm):** If you see a prewarm error, ensure your model paths in ComfyUI/models/Bonsai-4B/ are not empty and contain the necessary config.json files for each component.

* **VRAM Issues:** This model is highly optimized, but ensure you have at least 6GB of VRAM available to accommodate the loading process.

## ⚖️ License

This project uses the GpuPipeline architecture designed for isolated 1-bit/2-bit inference. Please ensure you comply with the licensing requirements of the underlying model weights provided by the original authors.
