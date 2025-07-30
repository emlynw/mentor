# -----------------------------------------------
#   sd_vae_decode.py   (save next to view script)
# -----------------------------------------------
import torch
import numpy as np
import cv2
from diffusers import AutoencoderKL
from transformers import CLIPFeatureExtractor

# --- initialise once -------------------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"   # same as wrapper
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16                               # wrapper stores fp16
RES      = 112                                         # whatever you used there
# latent shape per cam = (16, h, w) with h = w = RES//8
SIDE     = RES // 8

# VAE
_vae = (AutoencoderKL
        .from_pretrained(MODEL_ID, subfolder="vae")
        .to(DEVICE)
        .half()
        .eval())
for p in _vae.parameters():
    p.requires_grad = False

# CLIP mean / std so colours look right
_pre = CLIPFeatureExtractor.from_pretrained(
        "stabilityai/stable-diffusion-2",
        subfolder="feature_extractor")
_MEAN = torch.tensor(_pre.image_mean).view(1, 3, 1, 1).to(DEVICE)
_STD  = torch.tensor(_pre.image_std ).view(1, 3, 1, 1).to(DEVICE)

@torch.no_grad()
def latent_to_rgb(flat_lat: np.ndarray) -> np.ndarray:
    """
    flat_lat : 1‑D numpy array (16*SIDE*SIDE,) – fp16/32
    returns   : H×W×3 uint8  (BGR order for OpenCV)
    """
    lat = torch.as_tensor(flat_lat, device=DEVICE).reshape(
            1, 16, SIDE, SIDE).half()          # (1,16,h,w)

    # SD‑VAE decode() rescales internally, no 0.18215 needed
    img = _vae.decode(lat).sample               # (1,3,RES,RES) in [-1,1]
    img = (img * _STD + _MEAN).clamp(0, 1)      # de‑normalise
    img = (img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.resize(img, (720, 720))          # match your viewer
