import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from collections import deque
import cv2
import imageio
import os
from gymnasium.spaces import flatten_space, flatten
from copy import deepcopy

class PixelPairStack(gym.Wrapper):
    def __init__(self, env, image1_key='image1', image2_key='image2', stack_key='pixels'):
        super().__init__(env)
        self.image1_key = image1_key
        self.image2_key = image2_key
        self.stack_key = stack_key

        # Get the original image shape from the observation space.
        image_shape = env.observation_space[image1_key].shape
        # Remove a batch dimension if it exists.
        if len(image_shape) == 4:
            image_shape = image_shape[1:]
        # New observation will have shape (2, *original image shape*)
        stacked_shape = (2,) + image_shape
        self.observation_space[self.stack_key] = gym.spaces.Box(
            low=0, high=255, shape=stacked_shape, dtype=np.uint8
        )

    def _extract_image(self, obs, key):
        image = obs[key]
        # Remove batch dimension if present.
        if len(image.shape) == 4:
            image = image[0]
        return image.copy()  # Leave the image in its original format.

    def _transform_observation(self, obs):
        # Extract both images and stack them along a new axis (axis=0)
        img1 = self._extract_image(obs, self.image1_key)
        img2 = self._extract_image(obs, self.image2_key)
        obs[self.stack_key] = np.stack([img1, img2], axis=0)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_observation(obs), reward, terminated, truncated, info

class PixelFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='pixels'):
        super().__init__(env)
        self._num_frames = num_frames
        self.stack_key = stack_key
        self._frames = deque([], maxlen=num_frames)
        pixels_shape = env.observation_space[stack_key].shape
        

        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self.observation_space[stack_key] = Box(low=0, high=255, shape=(num_frames*pixels_shape[-1], *pixels_shape[:-1]), dtype=np.uint8)

    def _transform_observation(self, obs):
        assert len(self._frames) == self._num_frames
        obs[self.stack_key] = np.concatenate(list(self._frames), axis=0)
        return obs

    def _extract_pixels(self, obs):
        pixels = obs[self.stack_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pixels = self._extract_pixels(obs)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._extract_pixels(obs)
        self._frames.append(pixels)
        return self._transform_observation(obs), reward, terminated, truncated, info
    
class StateFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='state', flatten=True):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)   
        self.stack_key = stack_key
        self.flatten = flatten

        shape = self.env.observation_space[stack_key].shape
        if isinstance(shape, int):
            shape = (shape,)  # Convert to a tuple for consistency
        else:
            shape = shape  # If it's already a tuple or list, keep it as is
        if flatten: 
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames * shape[-1],), dtype=np.float32)
        else:
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames, *shape), dtype=np.float32)

    def _transform_observation(self):
        assert len(self._frames) == self._num_frames
        obs = np.array(self._frames)
        if self.flatten:
            obs = obs.flatten()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._num_frames):
            self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, reward, terminated, truncated, info

class CustomPixelObservation(gym.ObservationWrapper):
    """Resize (and optionally crop) a pixel observation."""

    def __init__(
        self,
        env: gym.Env,
        *,
        pixel_key: str = "pixels",
        crop_resolution=None,
        resize_resolution=None,
        save_old: bool = False,
        old_key: str = "old"
    ):
        super().__init__(env)

        # Handle int → (H, W) shorthand ----------------------------------------
        if isinstance(resize_resolution, int):
            resize_resolution = (resize_resolution, resize_resolution)
        if isinstance(crop_resolution, int):
            crop_resolution = (crop_resolution, crop_resolution)

        self.pixel_key = pixel_key
        self.save_old = save_old
        self.pixel_key_old = f"{pixel_key}_{old_key}" if save_old else None
        self.crop_resolution = crop_resolution
        self.resize_resolution = resize_resolution

        # ---------------------------------------------------------------------
        #  Build the new observation‑space
        # ---------------------------------------------------------------------
        orig_space = self.observation_space
        if not isinstance(orig_space, Dict):
            raise TypeError("Environment observation space must be gym.spaces.Dict")
        if pixel_key not in orig_space.spaces:
            raise KeyError(f"Expected '{pixel_key}' in observation dict")

        new_spaces = deepcopy(orig_space.spaces)
        orig_pix_space: Box = orig_space[pixel_key]
        orig_shape = orig_pix_space.shape
        n_channels = orig_shape[-1]

        # Shape after optional resize -----------------------------------------
        if resize_resolution is not None:
            new_shape = (*resize_resolution, n_channels)
        else:
            new_shape = orig_shape

        # Update processed image space
        new_spaces[pixel_key] = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        # Optionally add space for the original image -------------------------
        if save_old:
            new_spaces[self.pixel_key_old] = deepcopy(orig_pix_space)

        self.observation_space = Dict(new_spaces)

    # ---------------------------------------------------------------------
    #  Observation conversion
    # ---------------------------------------------------------------------
    def observation(self, observation):
        original_pixels = observation[self.pixel_key]

        # Preserve original if requested
        if self.save_old:
            observation[self.pixel_key_old] = original_pixels.copy()

        pixels = original_pixels  # work on a local reference

        # 1) Centre‑crop ------------------------------------------------------
        if self.crop_resolution is not None and pixels.shape[:2] != self.crop_resolution:
            h, w = pixels.shape[:2]
            crop_h, crop_w = self.crop_resolution
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
            pixels = pixels[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # 2) Resize -----------------------------------------------------------
        if self.resize_resolution is not None and pixels.shape[:2] != self.resize_resolution:
            pixels = cv2.resize(
                pixels,
                dsize=self.resize_resolution,
                interpolation=cv2.INTER_CUBIC,
            )

        observation[self.pixel_key] = pixels
        return observation
  
class VideoRecorder(gym.Wrapper):
    """Wrapper for rendering and saving rollouts to disk from a specific camera."""

    def __init__(
        self,
        env,
        save_dir,
        crop_resolution,
        resize_resolution,
        camera_name="wrist1",
        fps=10,
        current_episode=0,
        record_every=2,
        write_reward=False,
    ):
        super().__init__(env)

        self.save_dir = save_dir
        self.camera_name = camera_name
        os.makedirs(save_dir, exist_ok=True)
        num_vids = len([f for f in os.listdir(save_dir) if f.endswith(f"{camera_name}.mp4")])
        print(f"num_vids: {num_vids}")
        current_episode = num_vids * record_every

        if isinstance(resize_resolution, int):
            self.resize_resolution = (resize_resolution, resize_resolution)
        if isinstance(crop_resolution, int):
            self.crop_resolution = (crop_resolution, crop_resolution)

        self.resize_h, self.resize_w = self.resize_resolution
        self.crop_h, self.crop_w = self.crop_resolution
        self.fps = fps
        self.enabled = True
        self.current_episode = current_episode
        self.record_every = record_every
        self.write_reward = write_reward
        self.frames = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.current_episode % self.record_every == 0:
            frame = observation[self.camera_name].copy()

            # --------- add a blue channel if we only have RG ----------
            if frame.ndim == 3 and frame.shape[2] == 2:
                # OpenCV expects BGR, so prepend zeros = B
                b_plane = np.zeros_like(frame[..., 0:1])          # (H,W,1)
                frame   = np.concatenate([frame, b_plane], axis=-1)  # (H,W,3)
            
            if self.crop_resolution is not None:
                if frame.shape[:2] != (self.crop_h, self.crop_w):
                    center = frame.shape
                    x = center[1] // 2 - self.crop_w // 2
                    y = center[0] // 2 - self.crop_h // 2
                    frame = frame[int(y):int(y + self.crop_h), int(x):int(x + self.crop_w)]

            if self.resize_resolution is not None:
                if frame.shape[:2] != (self.resize_h, self.resize_w):
                    frame = cv2.resize(
                        frame,
                        dsize=(self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_CUBIC,
                    )
            frame  = np.ascontiguousarray(frame)
            if self.write_reward:
                cv2.putText(
                    frame,
                    f"{reward:.3f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            self.frames.append(frame)

        if terminated or truncated:
            if self.current_episode % self.record_every == 0:
                if info['success']:
                    filename = os.path.join(self.save_dir, f"{self.current_episode}_success_{self.camera_name}.mp4")
                else:
                    filename = os.path.join(self.save_dir, f"{self.current_episode}_failure_{self.camera_name}.mp4")
                imageio.mimsave(filename, self.frames, fps=self.fps)
                self.frames = []

            self.current_episode += 1

        return observation, reward, terminated, truncated, info
  
class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.
    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return observation, total_reward, terminated, truncated, info
  
class FrankaObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, camera_name='front'):
    super().__init__(env)
    self.camera_name = camera_name
    pixel_space = self.observation_space['images'][camera_name]
    self.state_keys = ['tcp_pose', 'gripper_pos']
    state_dim = 0
    for key in self.state_keys:
      state_dim += self.observation_space['state'][key].shape[0]
    state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
    self.observation_space = Dict({'pixels': pixel_space, 'state': state_space})
    
  def observation(self, observation):
    pixels = observation['images'][self.camera_name]
    state = np.concatenate([observation['state'][key] for key in self.state_keys])
    observation = {}
    observation['pixels'] = pixels
    observation['state'] = state
    return observation    
  

class DualCamObservation(gym.ObservationWrapper):
    """Stack two camera images along the channel dimension and leave the state unchanged."""
    def __init__(self, env, camera_name1='wrist1', camera_name2='wrist2'):
        super().__init__(env)
        self.camera_name1 = camera_name1
        self.camera_name2 = camera_name2

        # Get the pixel spaces from both cameras.
        pixel_space1 = self.observation_space[camera_name1]
        pixel_space2 = self.observation_space[camera_name2]

        # Extract shapes from each camera.
        h, w, c1 = pixel_space1.shape
        h2, w2, c2 = pixel_space2.shape
        assert h == h2 and w == w2, "Both cameras must have the same height and width."
        new_shape = (h, w, c1 + c2)

        # Concatenate the low and high bounds along the channel dimension.
        low = np.concatenate([pixel_space1.low, pixel_space2.low], axis=-1)
        high = np.concatenate([pixel_space1.high, pixel_space2.high], axis=-1)

        # Explicitly pass the new shape to the Box.
        stacked_space = Box(low=low, high=high, shape=new_shape, dtype=pixel_space1.dtype)

        # Leave the state observation space unchanged.
        state_space = self.observation_space['state']

        # Define the new observation space.
        self.observation_space = Dict({'pixels': stacked_space, 'state': state_space})

    def observation(self, observation):
        # Get images from both cameras.
        pixels1 = observation[self.camera_name1]
        pixels2 = observation[self.camera_name2]

        # Stack images along the channel dimension.
        pixels = np.concatenate([pixels1, pixels2], axis=-1)
        
        # Pass state through without modification.
        state = observation['state']
        
        return {'pixels': pixels, 'state': state}
  

  
# class FrankaDualCamObservation(gym.ObservationWrapper):
#   """Resize the observation to a given resolution"""
#   def __init__(self, env, camera1_name='wrist1', camera2_name='wrist2'):
#     super().__init__(env)
#     self.camera1_name = camera1_name
#     self.camera2_name = camera2_name
#     img1_space = self.observation_space['images'][camera1_name]
#     img2_space = self.observation_space['images'][camera2_name]
#     self.state_keys = ['panda/tcp_pos', 'panda/tcp_orientation', 'panda/gripper_pos', 'panda/gripper_vec']
#     state_dim = 0
#     for key in self.state_keys:
#       state_dim += self.observation_space['state'][key].shape[0]
#     state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
#     self.observation_space = Dict({'img1': img1_space, 'img2': img2_space, 'state': state_space})
    
#   def observation(self, observation):
#     img1 = observation['images'][self.camera1_name]
#     img2 = observation['images'][self.camera2_name]
#     state = np.concatenate([observation['state'][key] for key in self.state_keys])
#     observation = {}
#     observation['img1'] = img1
#     observation['img2'] = img2
#     observation['state'] = state
#     return observation  
  
class ActionState(gym.Wrapper):
    # Add previous action to the state
    def __init__(self, env, state_key='state', action_key='action'):
        super().__init__(env)
        self.action_key = action_key
        self.state_key = state_key
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space[state_key].shape[0]
        self.observation_space[state_key] = Box(low=-np.inf, high=np.inf, shape=(self.state_dim + self.action_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action = np.zeros(self.action_dim)
        obs[self.state_key] = np.concatenate([obs[self.state_key], self.action])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[self.state_key] = np.concatenate([obs[self.state_key], action])
        return obs, reward, terminated, truncated, info
    
class RotateImage(gym.ObservationWrapper):
    """Rotate the pixel observation by 180 degrees."""

    def __init__(self, env, pixel_key='pixels'):
        super().__init__(env)
        self.pixel_key = pixel_key

        # Optionally, update the observation space if needed.
        # Since a 180° rotation doesn't change the image shape,
        # we can just copy the existing space.
        self.observation_space = env.observation_space

    def observation(self, observation):
        # Extract the image from the observation using the specified key.
        image = observation[self.pixel_key]
        
         # Check if the image has a leading batch dimension.
        if image.shape[0] == 1:
            # Remove the batch dimension: shape becomes (height, width, 3)
            image = image[0]
            # Rotate the image by 180 degrees using OpenCV.
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            # Re-add the batch dimension: shape becomes (1, height, width, 3)
            rotated_image = np.expand_dims(rotated_image, axis=0)
        else:
            # Otherwise, just rotate the image normally.
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        
        
        # Replace the image in the observation with the rotated version.
        observation[self.pixel_key] = rotated_image
        return observation
    
class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images. separate privileged state space
    if it exists, for asymmetric RL.
    """

    def __init__(self, env, proprio_keys=None, priv_proprio_keys=None):
        super().__init__(env)

        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space.get("images", {})),
            }
        )

        self.has_priv_state = "priv_state" in self.env.observation_space.keys()
        if self.has_priv_state:
            self.priv_proprio_keys = priv_proprio_keys
            if self.priv_proprio_keys is None:
                self.priv_proprio_keys = list(self.env.observation_space["priv_state"].keys())

            self.priv_proprio_space = gym.spaces.Dict(
                {key: self.env.observation_space["priv_state"][key] for key in self.priv_proprio_keys}
            )
            self.observation_space["priv_state"] = flatten_space(self.priv_proprio_space)

    def observation(self, obs):
        new_obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs.get("images", {})),
        }
        if self.has_priv_state:
            new_obs["priv_state"] = flatten(
                self.priv_proprio_space,
                {key: obs["priv_state"][key] for key in self.priv_proprio_keys},
            )

        return new_obs

    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation.
        """
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs

class DoFConverterGymWrapper(gym.ActionWrapper):
    """
    Expose a smaller (3‑, 4‑, or 6‑DoF + gripper) action interface
    while the wrapped env still expects the full 7‑DoF command
        [dx, dy, dz, droll, dpitch, dyaw, dgrasp].

    Incoming action conventions
    ---------------------------
        ee_dof = 3 : [dx, dy, dz, grasp]
        ee_dof = 4 : [dx, dy, dz,  yaw, grasp]        (roll = pitch = 0)
        ee_dof = 6 : [dx, dy, dz, roll, pitch, yaw, grasp]

    Anything not supplied by the caller is filled with zeros before
    being forwarded to the underlying environment.
    """

    def __init__(self, env: gym.Env, ee_dof: int = 6):
        super().__init__(env)

        # ----- sanity checks -------------------------------------------------
        if not isinstance(env.action_space, Box) or env.action_space.shape[-1] != 7:
            raise ValueError(
                "DoFConverterGymWrapper assumes the wrapped env has a "
                "7‑dimensional continuous action space."
            )
        if ee_dof not in (3, 4, 6):
            raise ValueError("ee_dof must be 3, 4, or 6")
        self.ee_dof = ee_dof

        # ----- index mapping -------------------------------------------------
        # indices in the order the *wrapped* env expects
        if ee_dof == 3:          # (dx,dy,dz,grasp)
            self._map = [0, 1, 2, 6]
        elif ee_dof == 4:        # (dx,dy,dz,yaw,grasp)
            self._map = [0, 1, 2, 5, 6]
        else:                    # 6‑DoF   (full pose + grasp)
            self._map = [0, 1, 2, 3, 4, 5, 6]

        # ----- expose reduced action space ----------------------------------
        dim = len(self._map)                           # ee_dof + 1
        low  = self.env.action_space.low[self._map]
        high = self.env.action_space.high[self._map]
        print(f"DIM: {dim}, low: {low}, high: {high}")

        #  explicit shape for peace of mind
        self.action_space = Box(
            low   = low,
            high  = high,
            shape = (dim,),
            dtype = np.float32,
        )

    # ------------------------------------------------------------------ #
    #  convert small action → 7‑D before every env.step
    # ------------------------------------------------------------------ #
    def action(self, act: np.ndarray) -> np.ndarray:
        act = np.asarray(act, dtype=np.float32)
        if act.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {act.shape}"
            )

        full = np.zeros(7, dtype=np.float32)
        full[self._map] = act
        # print(f"action: {act}")
        # print(f"new action: {full}")
        return full

    # pretty‑print helper
    def __repr__(self):
        return f"DoFConverterGymWrapper(ee_dof={self.ee_dof})"
    
# -----------------------------------------------------------------------------#
#  Robomimic‑friendly SERL observation wrapper
# -----------------------------------------------------------------------------#
class SERLObsWrapperRobomimic(gym.ObservationWrapper):
    """
    ‑ For robomimic envs where *all* observations live in one dict.
    ‑ Separates `image_keys` out untouched;
      everything else is flattened into a single "state" vector.
    """

    def __init__(self, env, image_keys, proprio_keys=None):
        """
        Args
        ----
        env          : (gym.Env or compatible) a RobomimicGymWrapper instance.
        image_keys   : list[str]  keys whose values are image tensors (H,W,C)
        proprio_keys : list[str] or None
            Which non‑image keys to include in the flattened state.  If None,
            we use **all** keys that are *not* in `image_keys`.
        """
        super().__init__(env)

        self.image_keys = set(image_keys)

        # Split original observation space
        full_space = env.observation_space
        assert isinstance(full_space, gym.spaces.Dict)

        # ------------------- build proprio (non‑image) sub‑space -----------
        if proprio_keys is None:
            proprio_keys = [k for k in full_space.spaces.keys() if k not in self.image_keys]
        self.proprio_keys = proprio_keys

        self.proprio_space = gym.spaces.Dict(
            {k: deepcopy(full_space[k]) for k in self.proprio_keys}
        )

        # ------------------- new wrapped observation space -----------------
        img_spaces = {}
        for k in self.image_keys:
            chw_shape   = deepcopy(full_space[k]).shape
            hwc_shape   = self._chw_to_hwc_shape(chw_shape)            # <‑‑ only change
            img_spaces[k] = gym.spaces.Box(
                low   = 0,
                high  = 255,
                shape = hwc_shape,                                # H,W,C (or T,H,W,C)
                dtype = np.uint8,
            )

        self.observation_space = gym.spaces.Dict(
            {"state": flatten_space(self.proprio_space), **img_spaces}
        )

    def _chw_to_hwc_shape(self, shape):
        """
        (C,H,W)   -> (H,W,C)
        (T,C,H,W) -> (T,H,W,C)
        """
        if len(shape) == 3:          # C,H,W
            c, h, w = shape
            return (h, w, c)
        elif len(shape) == 4:        # T,C,H,W
            t, c, h, w = shape
            return (t, h, w, c)
        else:
            raise ValueError("unexpected image shape " + str(shape))


    # ------------------------------------------------------------------ #
    #  Observation conversion
    # ------------------------------------------------------------------ #
    def observation(self, obs):
        """
        Convert robomimic obs‑dict → dict{ "state": flat, img1, img2, … }.
        """
        state_dict = {k: obs[k] for k in self.proprio_keys}
        flat_state = flatten(self.proprio_space, state_dict)

        new_obs = {"state": flat_state}
        for k in self.image_keys:
            new_obs[k] = self._to_uint8_hwc(obs[k])
        return new_obs

    # Gymnasium’s ObservationWrapper already calls self.observation()
    # inside step / reset, but we want to preserve the two‑value reset API.
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def _to_uint8_hwc(self, arr):
        # (T,)C,H,W  → (T,)H,W,C  + cast to uint8
        if arr.ndim == 3:          # C,H,W
            arr = arr.transpose(1,2,0)
        elif arr.ndim == 4:        # T,C,H,W
            arr = arr.transpose(0,2,3,1)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        return arr

class RemoveBlueChannel(gym.ObservationWrapper):
    """
    Drop the blue channel (B) from every RGB image in `obs['images']`.

    ▸ Works with images shaped (H, W, C) or (1, H, W, C)  
    ▸ Preserves dtype, value range, and all other observation entries
    """

    def __init__(self, env: gym.Env, images_key: str = "images"):
        super().__init__(env)
        self.images_key = images_key

        # ---- rebuild the observation‑space with C‑>C‑1 --------------------
        orig_space = self.observation_space
        if images_key not in orig_space.spaces:
            raise KeyError(f"Expected '{images_key}' in observation dict")

        new_spaces = deepcopy(orig_space.spaces)
        img_space_dict = orig_space[images_key]
        if not isinstance(img_space_dict, Dict):
            raise TypeError(f"'{images_key}' must be a gym.spaces.Dict")

        # Adjust every camera sub‑space
        new_img_spaces = {}
        for cam, space in img_space_dict.spaces.items():
            h, w, c = space.shape[-3:]          # support (… H W C)
            if c < 3:
                raise ValueError(f"{cam} has <3 channels (shape {space.shape})")
            new_shape = (*space.shape[:-1], c - 1)  # drop blue
            new_img_spaces[cam] = Box(
                low  = space.low  [..., :2],       # keep RG bounds
                high = space.high [..., :2],
                shape=new_shape,
                dtype=space.dtype,
            )
        new_spaces[images_key] = Dict(new_img_spaces)

        # Final wrapped observation‑space
        self.observation_space = Dict(new_spaces)

    # ------------------------------------------------------------------ #
    #  Observation conversion
    # ------------------------------------------------------------------ #
    def observation(self, obs):
        imgs = obs[self.images_key]
        for cam, img in imgs.items():
            imgs[cam] = img[..., :2]
        return obs
    
class RemoveBlueChannelSingle(gym.ObservationWrapper):
    """
    Drop the blue channel (B) from a single RGB image stored in `obs[image_key]`.

    ▸ Works with images shaped (H, W, C) or (1, H, W, C)
    ▸ Preserves dtype, value range, and all other observation entries
    """

    def __init__(self, env: gym.Env, image_key: str = "image"):
        super().__init__(env)
        self.image_key = image_key

        # ---- rebuild the observation‑space with C‑>C‑1 --------------------
        orig_space = self.observation_space
        if image_key not in orig_space.spaces:
            raise KeyError(f"Expected '{image_key}' in observation dict")

        new_spaces = deepcopy(orig_space.spaces)
        img_space = orig_space[image_key]
        if not isinstance(img_space, Box):
            raise TypeError(f"'{image_key}' must be a gym.spaces.Box")

        # Adjust the image sub‑space
        h, w, c = img_space.shape[-3:]        # support (… H W C)
        if c < 3:
            raise ValueError(f"{image_key} has <3 channels (shape {img_space.shape})")
        new_shape = (*img_space.shape[:-1], c - 1)  # drop blue

        new_spaces[image_key] = Box(
            low  = img_space.low  [..., :2],   # keep RG bounds
            high = img_space.high [..., :2],
            shape=new_shape,
            dtype=img_space.dtype,
        )

        # Final wrapped observation‑space
        self.observation_space = Dict(new_spaces)

    # ------------------------------------------------------------------ #
    #  Observation conversion
    # ------------------------------------------------------------------ #
    def observation(self, obs):
        obs[self.image_key] = obs[self.image_key][..., :2]
        return obs

