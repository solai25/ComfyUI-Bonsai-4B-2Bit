"""Klein 4B inference forward (text -> PIL.Image) for the GPU/gemlite backend.

This is the glue between text encoder, transformer, and VAE. It mirrors the
upstream FLUX.2 inference path but strips out img2img, condition images, real
CFG, T-LoRA, and callback machinery that the GPU backend doesn't need.

Reused upstream helpers (from diffusers):
    Flux2Pipeline._prepare_text_ids                    # 4-axis RoPE for text tokens
    Flux2Pipeline._prepare_latent_ids                  # 4-axis RoPE for latent grid
    Flux2Pipeline._pack_latents                        # (B,C,H,W) -> (B,H*W,C)
    Flux2Pipeline._unpack_latents_with_ids             # scatter packed -> (B,C,H,W)
    Flux2Pipeline._unpatchify_latents                  # 2x2 unpack: (B,128,H,W) -> (B,32,2H,2W)
    retrieve_timesteps                                 # scheduler.set_timesteps wrapper

Empirical mu (resolution-dependent shift) is computed locally via
`_mflux_empirical_mu` to match the mflux + iOS Swift port byte-for-byte;
diffusers' built-in linear shift is NOT used.

Text encode is inlined locally (`_encode_klein_qwen3_prompt`) rather than reusing
`Flux2Pipeline._get_mistral_3_small_prompt_embeds` — that helper is the FLUX.2-dev
Mistral path (layers 10/20/30, Pixtral system message, `add_generation_prompt=False`)
and produces off-distribution embeddings for Klein/Qwen3.

Klein has `guidance_embeds=False`, so the inference path is single-forward
with a guidance scalar — no two-pass CFG. Default 4 steps.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps


log = logging.getLogger(__name__)


def _mflux_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Resolution-dependent shift from mflux. Mirrors
    mflux.models.common.schedulers.flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler._compute_empirical_mu
    (and the iOS port at apple/Bonsai/Pipeline/FlowMatchEulerScheduler.swift::computeEmpiricalMu).
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# Klein 4B: guidance_embeds=False, so the guidance scalar has no read path
# inside the transformer (Flux2TimestepGuidanceEmbeddings.forward short-circuits
# on `self.guidance_embedder is None`). Defaults: guidance=1.0, steps=4.
DEFAULT_GUIDANCE = 1.0
DEFAULT_NUM_STEPS = 4

# Klein/Qwen3 text-encoder layers stacked into the joint embedding.
KLEIN_OUTPUT_LAYERS = (9, 18, 27)


@torch.no_grad()
def _encode_klein_qwen3_prompt(
    text_encoder: nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_sequence_length: int,
) -> torch.Tensor:
    """Klein/Qwen3 prompt encode.

    No system message, plain string content, `add_generation_prompt=True`,
    `enable_thinking=False`, hidden states stacked from layers (9, 18, 27).
    Returns `(1, max_sequence_length, 3*hidden_dim)` in the encoder's dtype.
    """
    device = text_encoder.device
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    out = torch.stack([output.hidden_states[k] for k in KLEIN_OUTPUT_LAYERS], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)


def _build_default_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Default FLUX.2 flow-matching scheduler.

    The Klein model ships a `scheduler/scheduler_config.json` on HF; if the
    loader (coder8) wires `_scheduler` into GpuPipeline, prefer that. This
    fallback uses diffusers' defaults plus FLUX.2-style dynamic shift so
    the mflux empirical mu can flow through `set_timesteps(..., mu=mu)`.
    """
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )


@torch.no_grad()
def diffusion_forward(
    transformer: nn.Module,
    text_encoder: nn.Module,
    tokenizer,
    vae: nn.Module,
    prompt: str,
    *,
    height: int,
    width: int,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = 0,
    max_sequence_length: int = 512,
    guidance: float = DEFAULT_GUIDANCE,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
) -> Image.Image:
    """Klein 4B text-to-image forward (OPTIMIZED FOR 8GB VRAM)."""
    
    transformer_device = next(transformer.parameters()).device

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"height={height} and width={width} must be multiples of 32 (vae_scale_factor*2).")

    if scheduler is None:
        scheduler = _build_default_scheduler()

    # ==========================================
    # STEP 1: TEXT ENCODING (Move to GPU, then immediately Offload)
    # ==========================================
    log.info("encoding prompt (max_seq=%d, klein/qwen3)", max_sequence_length)
    
    # Move Text Encoder to GPU just in time
    text_encoder.to(transformer_device)
    
    prompt_embeds = _encode_klein_qwen3_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
    )
    text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(transformer_device)
    
    # 💥 MASSIVE VRAM SAVINGS: Offload Text Encoder to CPU and clear cache
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    # ==========================================

    activation_dtype = getattr(transformer, "_inference_dtype", torch.float16)
    prompt_embeds_t = prompt_embeds.to(device=transformer_device, dtype=activation_dtype)

    # 2. Prepare initial latents
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    in_channels_latents = transformer.config.in_channels // 4

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_shape = (1, in_channels_latents * 4, h_lat // 2, w_lat // 2)
    latents_4d = torch.randn(noise_shape, generator=gen, dtype=torch.float32)
    latents_4d = latents_4d.to(device=transformer_device, dtype=activation_dtype)

    latent_ids = Flux2Pipeline._prepare_latent_ids(latents_4d).to(transformer_device)
    latents = Flux2Pipeline._pack_latents(latents_4d)
    image_seq_len = latents.shape[1]
    
    log.info("latents: 4D=%s -> packed=%s  image_seq_len=%d", tuple(latents_4d.shape), tuple(latents.shape), image_seq_len)

    # 3. Schedule timesteps
    mu = _mflux_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    timesteps, num_steps_eff = retrieve_timesteps(
        scheduler, num_steps, transformer_device, sigmas=sigmas, mu=mu,
    )
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
        
    log.info("scheduling: num_steps=%d mu=%.4f", num_steps_eff, mu)
    guidance_t = torch.full([1], guidance, device=transformer_device, dtype=torch.float32).expand(latents.shape[0])

    # 4. Denoising loop
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance_t,
            encoder_hidden_states=prompt_embeds_t,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]

        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    # ==========================================
    # STEP 5: VAE DECODING (Move to GPU, Decode, then Offload)
    # ==========================================
    # Move VAE to GPU only when denoising is completely finished
    vae.to(transformer_device)
    vae_device = transformer_device

    latents = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids)
    latents = latents.to(device=vae_device, dtype=torch.bfloat16)

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean
    latents = Flux2Pipeline._unpatchify_latents(latents)

    image = vae.decode(latents, return_dict=False)[0]
    
    # 💥 MORE VRAM SAVINGS: Offload VAE back to CPU so the GPU is clean for the next generation
    vae.to("cpu")
    torch.cuda.empty_cache()
    # ==========================================

    # 6. Tensor -> PIL
    img = image[0].clamp(-1.0, 1.0).float()
    img = (img + 1.0) * 127.5
    img = img.clamp(0.0, 255.0).round().to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    
    return Image.fromarray(img, mode="RGB")


__all__ = ["diffusion_forward", "DEFAULT_GUIDANCE", "DEFAULT_NUM_STEPS"]
