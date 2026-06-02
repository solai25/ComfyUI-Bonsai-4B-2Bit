ComfyUI-Bonsai-4B

A custom ComfyUI node to run the Bonsai 4B (2-bit Ternary) image generation model locally. This node utilizes optimized Gemlite and HQQ kernels for high-speed inference on Windows/Linux GPU environments.
⚠️ Prerequisites

This node requires specific 2-bit/4-bit quantization libraries. Before installing, ensure you have the following in your ComfyUI environment:
Bash

```
pip install hqq gemlite triton-windows
```

## 🛠️ Installation

    Navigate to your ComfyUI/custom_nodes/ directory.

    Clone this repository:
    Bash

    git clone https://github.com/solai25/ComfyUI-Bonsai-4B-2Bit

    (Optional) If you don't use Git, download the folder and place it in your custom_nodes/ directory.

## 📥 Model Setup

The node expects the model files to be organized in your ComfyUI directory:

'''
ComfyUI/
└── models/
    └── Bonsai-4B/   <-- Create this folder
        ├── text_encoder-hqq-4bit/
        ├── transformer-gemlite-int2/
        └── vae/
'''

Note: If the model is not found, the node will attempt to automatically download the required weights from Hugging Face upon the first run.
## 🚀 Usage

    Restart ComfyUI.

    Right-click the canvas and select Add Node > Bonsai Image > Bonsai 4B (Gemlite 2-bit).

    Connect the IMAGE output to a Save Image node.

    Configure your prompt and inference settings.

    Queue Prompt. The model will warm up on the GPU for the first generation.

## 🧩 Node Structure
## 📝 Troubleshooting

    Missing Kernels: Ensure triton is installed correctly for your specific GPU architecture.

    RuntimeError (prewarm): If you see a prewarm error, ensure your model paths in ComfyUI/models/Bonsai-4B/ are not empty and contain the necessary config.json files for each component.

    VRAM Issues: This model is highly optimized, but ensure you have at least 6GB of VRAM available to accommodate the loading process.

## ⚖️ License

This project uses the GpuPipeline architecture designed for isolated 2-bit inference. Please ensure you comply with the licensing requirements of the underlying model weights provided by the original authors.
