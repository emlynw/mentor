# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Replay buffer for storing image states and state states seperately
# Taken from drqv2 

import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0]


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, replay_dir, frame_stack):
        self._replay_dir = replay_dir
        self._frame_stack = frame_stack
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        if time_step['first']:
            for i in range(self._frame_stack):
                for name, value in time_step.items():
                        self._current_episode[name].append(value)
        else:
            for name, value in time_step.items():
                self._current_episode[name].append(value)
        if time_step['truncated'] or time_step['terminated']:
            episode = dict()
            for name, value in time_step.items():
                value = self._current_episode[name]
                episode[name] = np.array(value)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += (int(eps_len) - (self._frame_stack-1))

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode) - (self._frame_stack-1)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, frame_stack, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, sample_key):
        self._replay_dir = replay_dir
        self._frame_stack = frame_stack
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.sample_key = sample_key

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len - (self._frame_stack-1) + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= (episode_len(early_eps)-(self._frame_stack-1))
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += (eps_len-(self._frame_stack-1))

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + (eps_len-(self._frame_stack-1)) > self._max_size:
                break
            fetched_size += (eps_len-(self._frame_stack-1))
            # if not self._store_episode(eps_fn):
            #     break
            if not self._store_episode(eps_fn):
                print("Warning: could not load", eps_fn, "-- skipping.")
                continue

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx = np.random.randint(low=self._frame_stack, high=episode_len(episode)-self._nstep+1)
        img = episode[self.sample_key][idx-self._frame_stack:idx]
        img = img.reshape(img.shape[0]*img.shape[1], *img.shape[2:])
        next_img = episode[self.sample_key][idx+self._nstep-self._frame_stack:idx+self._nstep]
        next_img = next_img.reshape(next_img.shape[0]*next_img.shape[1], *next_img.shape[2:])
        state = episode['state'][idx-1]
        next_state = episode['state'][idx+self._nstep-1]
        action = episode['action'][idx]
        if episode['terminated'][idx]:
            mask = 0.0
        else:
            mask = 1.0
        mask = np.float32(mask)
        mask = np.expand_dims(mask, axis=0)
        reward = 0.0
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward = reward + self._discount**i * step_reward
        reward = reward.astype(np.float32)
        action = action.astype(np.float32)
        reward = np.expand_dims(reward, axis=0)
        return (img, state, action, reward, mask, next_img, next_state)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = int(np.random.get_state()[1][0] + worker_id)
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, frame_stack, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, sample_key):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            frame_stack,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            sample_key=sample_key)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader