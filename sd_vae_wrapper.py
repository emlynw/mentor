# -------------------------------------------------------------
# SD-3.5 VAE wrapper  +  Random-Shift aug  +  CLIP preprocessing
# -------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPFeatureExtractor
import cv2
import time


# ---------- DrQ-v2 random-shift module ------------
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


# ---------- Wrapper -------------------------------------------
class SDVAEWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 augment: bool = False,
                 res: int = 112,
                 pad: int = 4,
                 image_key: str = "pixels",
                 half: bool = True,
                 compile: bool = True,
                 model_id: str = "stabilityai/stable-diffusion-3.5-large",
                 token: str | None = None):
        super().__init__(env)

        # ---------- device / dtype ----------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if half else torch.float32
        torch.backends.cudnn.benchmark = True

        # ---------- VAE ----------
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", token=token).to(self.device)
        if half:
            vae = vae.half()
        if compile:
            vae = torch.compile(vae, mode="reduce-overhead", fullgraph=True)
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # ---------- CLIP image pre-processor ----------
        self.preproc = CLIPFeatureExtractor.from_pretrained("stabilityai/stable-diffusion-2", subfolder="feature_extractor")
        self.preproc.size      = {"shortest_edge": res}
        self.preproc.crop_size = {"height": res, "width": res}

        self.augment   = augment
        self.rs_aug    = RandomShiftsAug(pad)
        self.res       = res
        self.half      = half
        self.image_key = image_key

        # ---------- observation space ----------
        h_lat = res // 8
        w_lat = res // 8
        flat  = 16* h_lat * w_lat
        n_lat = 2 if augment else 1

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise TypeError("Env observation must be gym.spaces.Dict")

        spaces = self.observation_space.spaces.copy()
        spaces["embedding"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_lat, flat),
            dtype=np.float16 if half else np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

        # ---------- one warm-up encode ----------
        dummy = torch.zeros(1, 3, res, res,
                            dtype=self.dtype, device=self.device)
        with torch.no_grad():
            _ = self.vae.encode(dummy)


    # --------------- main hook -----------------------------------
    def observation(self, observation):
        start_time = time.time()
        pil = Image.fromarray(observation[self.image_key])
        processed = self.preproc(images=pil, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        if self.augment:
            aug_processed = self.rs_aug(processed.clone())
            img_tensors = torch.cat([processed, aug_processed], 0)  # BCHW
        else:
            img_tensors = processed
        if self.half:
            img_tensors = img_tensors.half()  # BCHW fp16

        with torch.no_grad():
            post = self.vae.encode(img_tensors)
            lat  = post.latent_dist.mean

        # mean = torch.tensor(self.preproc.image_mean).view(1, 1, 3)
        # std = torch.tensor(self.preproc.image_std).view(1, 1, 3)
        
        # # decode image
        # with torch.no_grad():
        #     outputs = self.vae.decode(lat)
        # decoded_image = outputs.sample[1]
        # decoded_image_processed = decoded_image.squeeze(0).cpu().permute(1, 2, 0)
        # decoded_image_processed = decoded_image_processed * std + mean
        # decoded_image_processed = decoded_image_processed.clamp(0, 1).numpy()
        # #resize to 640
        # decoded_image_processed = cv2.resize(decoded_image_processed, (640, 640), interpolation=cv2.INTER_LINEAR)
        # decoded_image_processed = cv2.cvtColor(decoded_image_processed, cv2.COLOR_RGB2BGR)
        # cv2.imshow("decoded_image", decoded_image_processed)
        # cv2.waitKey(1)

        observation["embedding"] = lat.flatten(1).cpu().numpy()  # (1, flat_dim)
        return observation
    
class SDVAEWrapperDualCam(gym.ObservationWrapper):
    """
    Dual-camera wrapper: processes two cameras, optionally augments via DrQ random-shifts,
    and encodes all in one batch through Stable-Diffusion VAE.

    observation[wrist1_key], observation[image2_key]: each H×W×3 uint8 arrays
    Outputs:
      - 'embedding_norm': concatenated latents of wrist1 and image2 (no aug)
      - 'embedding_aug' : concatenated latents after random shifts (if augment=True)
    Both are 1D arrays of length 2*flat_dim.
    """
    def __init__(
        self,
        env: gym.Env,
        augment: bool = False,
        res: int = 112,
        pad: int = 4,
        image1_key: str = "image1",
        image2_key: str = "image2",
        half: bool = True,
        compile: bool = True,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        token: str | None = None,
    ):
        super().__init__(env)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if half else torch.float32
        torch.backends.cudnn.benchmark = True

        # Load and prepare VAE
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", token=token).to(self.device)
        if half:
            vae = vae.half()
        if compile:
            vae = torch.compile(vae, mode="reduce-overhead", fullgraph=True)
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # CLIP preprocessing
        self.preproc = CLIPFeatureExtractor.from_pretrained(
            "stabilityai/stable-diffusion-2", subfolder="feature_extractor"
        )
        self.preproc.size      = {"shortest_edge": res}
        self.preproc.crop_size = {"height": res, "width": res}

        self.augment      = augment
        self.rs_aug       = RandomShiftsAug(pad)
        self.res          = res
        self.image1_key   = image1_key
        self.image2_key   = image2_key
        self.half         = half

        # Compute flat dim
        h_lat = res // 8
        w_lat = res // 8
        self.flat = 16 * h_lat * w_lat

        # Update observation space
        assert isinstance(self.observation_space, gym.spaces.Dict), "Obs space must be Dict"
        spaces = self.observation_space.spaces.copy()
        spaces['embedding'] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, 2 * self.flat,),
            dtype=(np.float16 if half else np.float32),
        )
        if augment:
            spaces['embedding'] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(2, 2 * self.flat,),
                dtype=(np.float16 if half else np.float32),
            )
        self.observation_space = gym.spaces.Dict(spaces)

        # Warm-up
        dummy = torch.zeros(1, 3, res, res, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            _ = self.vae.encode(dummy)

    def observation(self, observation):
        start_time = time.time()
        # Load two camera images
        pil1 = Image.fromarray(observation[self.image1_key]).convert('RGB')
        pil2 = Image.fromarray(observation[self.image2_key]).convert('RGB')
        # Preprocess both
        tensor_batch = self.preproc(images=[pil1, pil2], return_tensors='pt').pixel_values.to(self.device)
        if self.half:
            tensor_batch = tensor_batch.half()

        # If augment, append shifted variants
        if self.augment:
            aug_batch = self.rs_aug(tensor_batch)
            full_batch = torch.cat([tensor_batch, aug_batch], dim=0)  # (4,3,res,res)
        else:
            full_batch = tensor_batch  # (2,3,res,res)

        # Single VAE encode
        with torch.no_grad():
            post = self.vae.encode(full_batch)
            lat = post.latent_dist.mean  # shape (N,16,h_lat,w_lat)

        # mean = torch.tensor(self.preproc.image_mean).view(1, 1, 3)
        # std = torch.tensor(self.preproc.image_std).view(1, 1, 3)
        #  # decode image
        # with torch.no_grad():
        #     outputs = self.vae.decode(lat)
        # decoded_image = outputs.sample[3]
        # decoded_image_processed = decoded_image.squeeze(0).cpu().permute(1, 2, 0)
        # decoded_image_processed = decoded_image_processed * std + mean
        # decoded_image_processed = decoded_image_processed.clamp(0, 1).numpy()
        # #resize to 640
        # decoded_image_processed = cv2.resize(decoded_image_processed, (640, 640), interpolation=cv2.INTER_LINEAR)
        # decoded_image_processed = cv2.cvtColor(decoded_image_processed, cv2.COLOR_RGB2BGR)
        # cv2.imshow("decoded_image", decoded_image_processed)
        # cv2.waitKey(1)

        flat = lat.flatten(1)  # (N, flat)

        # Norm embedding
        norm0, norm1 = flat[0], flat[1]
        embs = torch.cat([norm0, norm1], dim=-1).unsqueeze(0)  # (1, flat_dim)

        # Aug embedding if requested
        if self.augment:
            aug0, aug1 = flat[2], flat[3]
            emb_aug = torch.cat([aug0, aug1], dim=-1).unsqueeze(0)  # (1, flat_dim)
            embs = torch.cat([embs, emb_aug], dim=0)

        observation["embedding"] = embs.cpu().numpy()
        return observation

