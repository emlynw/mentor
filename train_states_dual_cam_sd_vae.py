import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import wandb

import gymnasium as gym
import fruit_gym
from datetime import datetime
from logger import Logger
import omegaconf


from wrappers import ActionRepeat, VideoRecorder, CustomPixelObservation, PixelFrameStack, PixelPairStack, StateFrameStack, ActionState, RotateImage, SERLObsWrapper
from sd_vae_wrapper import SDVAEWrapperDualCam
from gymnasium.wrappers import TimeLimit
from replay_buffer_states import ReplayBufferStorage, make_replay_loader
import wandb
import re
import cv2

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = int(obs_spec['embedding'].shape[0]*obs_spec['embedding'].shape[-1])
    cfg.state_dim = int(np.prod(np.array([obs_spec['state'].shape])))
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.setup()

        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            wandb.init(project="Mentor", group=group_name,name=exp_name)
            
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.results_dir = os.path.join(self.work_dir, 'results', self.start_time)
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        
        # create envs
        self.train_env = self.create_environment(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat, record=self.cfg.save_train_video, 
                                                 vid_folder='train videos', state_res=self.cfg.state_res, video_res=self.cfg.video_res)
        self.eval_env = self.create_environment(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat, record=self.cfg.save_video, 
                                                vid_folder='eval videos', state_res=self.cfg.state_res, video_res=self.cfg.video_res)
        print(f"embedding shape: {self.train_env.observation_space['embedding'].shape}")
        print(f"state shape: {self.train_env.observation_space['state'].shape}")
        print(f"action shape: {self.train_env.action_space.shape}")
        
        # Create agent
        self.agent = make_agent(self.train_env.observation_space, self.train_env.action_space, self.cfg.agent)
        
        # create replay buffer
        self.replay_storage = ReplayBufferStorage(self.work_dir / "buffer", self.cfg.frame_stack)
        buffer_dir = (self.work_dir / "buffer").resolve()
        print("Expecting buffer in:", buffer_dir, " – files:", len(list(buffer_dir.iterdir())))

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer", 
            self.cfg.frame_stack, 
            self.cfg.replay_buffer_size,
            self.cfg.batch_size * self.cfg.utd,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_buffer,
            self.cfg.nstep,
            self.cfg.discount,
            'emb',)
        self._replay_iter = None
        self.max_reward = -np.inf
    
    def create_environment(self, name, frame_stack=1, action_repeat=2, record=False, vid_folder='eval videos', state_res=128, video_res=224):
        proprio_keys = ["tcp_pose", "gripper_pos"]
        cameras = ["wrist1", "wrist2"]
        print(f"augment: {self.cfg.augment}")
        if self.cfg.augment:
            self.i = 1
            env = gym.make(name, render_mode='rgb_array', ee_dof = 6, cameras=cameras, reward_type="dense", height=self.cfg.aug_res, width=self.cfg.aug_res, gripper_pause=True)
        else:
            self.i = 0
            env = gym.make(name, render_mode='rgb_array', ee_dof = 6, cameras=cameras, reward_type="dense", height=self.cfg.video_res, width=self.cfg.video_res, gripper_pause=True)
        video_dir=os.path.join(self.work_dir, vid_folder)
        if action_repeat > 1:
            env = ActionRepeat(env, action_repeat)
        env = TimeLimit(env, max_episode_steps=self.cfg.max_episode_steps)
        env = SERLObsWrapper(env, proprio_keys=proprio_keys)
        env = RotateImage(env, pixel_key="wrist1")
        if record:
            for image_name in cameras:
                    crop_res = env.observation_space[image_name].shape[0]
                    env = VideoRecorder(env, video_dir, camera_name=image_name, crop_resolution=crop_res, resize_resolution=video_res, fps=20, record_every=2, write_reward=True)
        env = ActionState(env)
        if self.cfg.augment:
            env = CustomPixelObservation(env, pixel_key="wrist1", crop_resolution=self.cfg.aug_res, resize_resolution=self.cfg.aug_res)
            env = CustomPixelObservation(env, pixel_key="wrist2", crop_resolution=self.cfg.aug_res, resize_resolution=self.cfg.aug_res)
        else:
            env = CustomPixelObservation(env, pixel_key="wrist1", crop_resolution=video_res, resize_resolution=state_res)
            env = CustomPixelObservation(env, pixel_key="wrist2", crop_resolution=video_res, resize_resolution=state_res)
        env = SDVAEWrapperDualCam(env, augment=self.cfg.augment, res=state_res, pad=4, image1_key="wrist1", image2_key="wrist2")
        env = StateFrameStack(env, frame_stack, stack_key='embedding', flatten=False)
        env = StateFrameStack(env, frame_stack, stack_key='state')
        return env

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def scale_action(self, a):
        low, high = self.train_env.action_space.low, self.train_env.action_space.high
        range = high - low
        a = (range/2.0)*(a+1.0) + low
        return a
    
    def eval(self):
        step, episode, total_reward, end_reward = 0, 0, 0, 0
        successes = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            obs, info = self.eval_env.reset()
            # Make sure all data is float32
            emb = obs['embedding'].astype(np.float32)
            state = obs['state'].astype(np.float32)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(emb[:,0].flatten(), self.global_step, True, obs_sensor=state)
                    action = self.scale_action(action).astype(np.float32)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                emb = obs['embedding'].astype(np.float32)
                state = obs['state'].astype(np.float32)
                total_reward += reward
                step += 1
            end_reward += reward
            if info['success']:
                successes += 1
            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('success_rate', successes / episode)
            log('episode_reward', total_reward / episode)
            log('episode_end_reward', end_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)  

        return total_reward / episode

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        obs, info = self.train_env.reset()
        emb = obs['embedding'].astype(np.float32)
        state = obs['state'].astype(np.float32)
        action = self.train_env.action_space.sample().astype(np.float32)
        reward = np.float32(0.0)
        terminated = False
        truncated = False
        first = True
        time_step = {"emb": emb[-1, self.i], "state": state, "action": action, "reward": reward, "first": first, "terminated": terminated, "truncated": truncated}
        self.replay_storage.add(time_step)
        metrics = None
        while train_until_step(self.global_step):
            if terminated or truncated:
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_end_reward', reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                obs, info = self.train_env.reset()
                emb = obs['embedding'].astype(np.float32)
                state = obs['state'].astype(np.float32)
                first = True
                terminated = False
                truncated = False
                reward = np.float32(0.0)
                time_step = {"emb": emb[-1, self.i], "state": state, "action": action, "reward": reward, "first": first, "terminated": terminated, "truncated": truncated}
                self.replay_storage.add(time_step)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                eval_reward = self.eval()
                if self.cfg.save_model:
                    self.save_model()
                    if eval_reward > self.max_reward:
                        self.max_reward = eval_reward
                        self.save_model('snapshot_best.pt')


            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(emb[:,0].flatten(), self.global_step, False, obs_sensor=state)
                action = self.scale_action(action).astype(np.float32)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self.cfg.update_every_steps == 0 else dict(
                )
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            ## if finetuning with no data, uncomment the following lines
            # if not seed_until_step(self.global_step):
            #     if self.global_step - self.load_step < 20000:
            #         pass
            #     elif self.global_step - self.load_step == 20000:
            #         print(f"starting training")
            #         metrics = self.agent.update(
            #             self.replay_iter, self.global_step
            #         ) if self.global_step % self.cfg.update_every_steps == 0 else dict(
            #         )
            #         self.logger.log_metrics(metrics, self.global_frame, ty='train')
            #     else:
            #         metrics = self.agent.update(
            #             self.replay_iter, self.global_step
            #         ) if self.global_step % self.cfg.update_every_steps == 0 else dict(
            #         )
            #         self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            obs, reward, terminated, truncated , info = self.train_env.step(action)
            emb = obs['embedding'].astype(np.float32)
            state = obs['state'].astype(np.float32)
            first = False
            reward = np.float32(reward)
            episode_reward += reward
            time_step = {"emb": emb[-1, self.i], "state": state, "action": action, "reward": reward, "first": first, "terminated": terminated, "truncated": truncated}
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_model(self, name="snapshot.pt"):
        snapshot = self.work_dir / name
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v
        self.load_step = self._global_step


@hydra.main(config_path='cfgs', config_name='config_sdvae_dual_cam')
def main(cfgs):
    from train_states_dual_cam_sd_vae import Workspace as W
    root_dir = Path.cwd()
    print(f"root_dir: {root_dir}")
    workspace = W(cfgs)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
        print(f"resuming from global step: {workspace.global_step}")
    workspace.train()


if __name__ == '__main__':
    main()