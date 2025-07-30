import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models import Resnet18LinearEncoderNet
import gymnasium as gym
import numpy as np
import pickle

class DistanceShapeRewardWrapper(gym.Wrapper):
    """
    R' = R_sparse + γ * (d_prev - d_next)
    where d = || emb(s) - emb(goal) ||.
    """

    def __init__(
        self,
        env,
        model,
        goal_emb,
        distance_scale,
        image_key="wrist1",
        device: str = None,
        alpha: float = 10.0
    ):
        super().__init__(env)
        self.image_key = image_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.goal_emb = goal_emb
        self.distance_scale = distance_scale
        self.model = model


    def step(self, action):
        obs, sparse_r, terminated, truncated, info = self.env.step(action)

        # compute next distance
        d_next = self._compute_dist(obs)
        # shaping term = γ * (d_prev - d_next)
        shaping = -self.distance_scale * d_next
        # new reward
        r_prime = self.alpha*(sparse_r + shaping)

        # log everything
        info["orig_reward"]  = sparse_r
        info["shaping_term"] = float(shaping)

        # update for next step
        self.prev_dist = d_next
        return obs, r_prime, terminated, truncated, info

    def _compute_dist(self, obs):
        # grab image array
        im = obs[self.image_key]
        # unwrap time/batch if present
        if im.ndim == 4:
            im = im[0]
        # normalize to [0,1], permute to (B=1,C,H,W)
        im_t = (
            torch.from_numpy(im.astype(np.float32) / 255.0)
                 .permute(2,0,1)
                 .unsqueeze(0).unsqueeze(0)
                 .to(self.device)
        )
        with torch.no_grad():
            emb = self.model.infer(im_t).embs.to(self.device).squeeze(0)  # (D,)
        # compute L2 distance to goal embedding
        dist = torch.norm(emb - self.goal_emb, p=2).item()
        return dist

class DistanceShapeRewardWrapperDelta(gym.Wrapper):
    """
    R' = R_sparse + γ * (d_prev - d_next)
    where d = || emb(s) - emb(goal) ||.
    """

    def __init__(
        self,
        env,
        model,
        goal_emb,
        distance_scale,
        image_key="wrist1",
        device: str = None,
        alpha: float = 10.0
    ):
        super().__init__(env)
        self.image_key = image_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.goal_emb = goal_emb
        self.distance_scale = distance_scale
        self.model = model
        self.prev_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_dist = self._compute_dist(obs)
        return obs, info

    def step(self, action):
        obs, sparse_r, terminated, truncated, info = self.env.step(action)

        # compute next distance
        d_next = self._compute_dist(obs)
        # shaping term = γ * (d_prev - d_next)
        shaping = self.distance_scale * (self.prev_dist - d_next)
        # new reward
        r_prime = self.alpha*(sparse_r + shaping)

        # log everything
        info["orig_reward"]  = sparse_r
        info["dist_prev"]    = float(self.prev_dist)
        info["dist_next"]    = float(d_next)
        info["shaping_term"] = float(shaping)

        # update for next step
        self.prev_dist = d_next
        return obs, r_prime, terminated, truncated, info

    def _compute_dist(self, obs):
        # grab image array
        im = obs[self.image_key]
        # unwrap time/batch if present
        if im.ndim == 4:
            im = im[0]
        # normalize to [0,1], permute to (B=1,C,H,W)
        im_t = (
            torch.from_numpy(im.astype(np.float32) / 255.0)
                 .permute(2,0,1)
                 .unsqueeze(0).unsqueeze(0)
                 .to(self.device)
        )
        with torch.no_grad():
            emb = self.model.infer(im_t).embs.to(self.device).squeeze(0)  # (D,)
        # compute L2 distance to goal embedding
        dist = torch.norm(emb - self.goal_emb, p=2).item()
        return dist

class xirlResnet18RewardWrapper(gym.Wrapper):
    def __init__(self, env, image_key="wrist1", device=None):
        super().__init__(env)
        self.image_key = image_key
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        goal_emb_path = "/home/emlyn/xirl_results/pretrain_runs/128_pull_anchors_copy/goal_emb.pkl"
        distance_scale_path = "/home/emlyn/xirl_results/pretrain_runs/128_pull_anchors_copy/distance_scale.pkl"
        xirl_resnet_18 = torch.load('/home/emlyn/xirl_results/pretrain_runs/128_pull_anchors_copy/checkpoints/1518.ckpt')
        with open(goal_emb_path, "rb") as fp:
            self.goal_emb = pickle.load(fp)
        with open(distance_scale_path, "rb") as fp:
            self.distance_scale = pickle.load(fp)
        model = Resnet18LinearEncoderNet(embedding_size=128, num_ctx_frames=1,
                                normalize_embeddings=False, learnable_temp=False)
        model.load_state_dict(xirl_resnet_18['model'])
        model.to(self.device).eval()
        self.model = model
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        orig_reward = reward
        pixels = obs[self.image_key].copy()
        pixels = np.transpose(pixels, (2, 0, 1))
        pixels_shape = pixels.shape
        pixels = torch.from_numpy(pixels.reshape(1 ,1 ,*pixels_shape)).float()
        pixels = pixels / 255.0
        pixels = pixels.to(self.device)
        with torch.no_grad():
            obs_emb = self.model.infer(pixels).embs
        obs_emb = obs_emb.cpu().numpy()
        dist = np.linalg.norm(obs_emb - self.goal_emb)
        reward = -dist * self.distance_scale
        reward = (reward+1.5)**4
        info['orig_reward'] = orig_reward
        return obs, reward, terminated, truncated, info

