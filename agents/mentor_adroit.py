import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
import utils

from moe import MoE
import numpy as np
import torchvision.models as models

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


class Encoder(nn.Module):
    def __init__(self, obs_shape, encoder_type, resnet_fix, pretrained_factor): # encoder_type in ['scratch', 'spawnnet']
        super().__init__()

        assert len(obs_shape) == 3

        self.encoder_type = encoder_type
        self.pretrained_factor = pretrained_factor
        
        assert encoder_type in ['scratch', 'spawnnet']
                
        if self.encoder_type == 'scratch':
            self.repr_dim = 32 * 35 * 35
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU())
            self.apply(utils.weight_init)
        
        if self.encoder_type == 'spawnnet':
            self.pretrained_resnet = models.resnet18(pretrained=True)
            if resnet_fix:
                for params in self.pretrained_resnet.parameters():
                    params.requires_grad = False
            # pretrained shape: 64 * 21 * 21 -> 128 * 11 * 11
            
            self.scratch_convnet_layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), # shape: 84 * 84 -> 42 * 42
                                        nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), # shape: 42 * 42 -> 21 * 21
                                        )
            self.scratch_convnet_layer2 = nn.Sequential(nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1), # shape: 21 * 21 -> 21 * 21
                                        nn.ReLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), # shape: 21 * 21 -> 11 * 11
                                        )
            # scratch shape: 32 * 21 * 21 -> 64 * 11 * 11
            self.oneXone_conv_layer1 = nn.Sequential(nn.Conv2d(64, 32, 1, stride=1))
            self.oneXone_conv_layer2 = nn.Sequential(nn.Conv2d(128, 64, 1, stride=1))
            self.residual_conv_layer1 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1),
                                               nn.ReLU(), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU())
            self.residual_conv_layer2 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                               nn.ReLU(), nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU())
            
            feature_dim = obs_shape[0]//9 * 4 * 128 * 11 * 11
            self.repr_dim = 1024
            self.output_layer = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, self.repr_dim))
            self.apply(utils.weight_init)

    def forward(self, x):
        x = x / 255.0 - 0.5
        
        if self.encoder_type == 'scratch':
            h = self.convnet(x) # shape: d * h * w
            h = h.view(h.shape[0], -1)
            return h

        if self.encoder_type == 'spawnnet':
            bsz = x.shape[0]
            n_camera = x.shape[1] // 9
            x = x.view(-1, 3, 3, x.shape[2], x.shape[3]).view(-1, 3, x.shape[2], x.shape[3])
            
            hidden_pretrained = x.detach()
            hidden_scratch = x
            
            # Layer 1
            with torch.no_grad():
                for name, module in self.pretrained_resnet._modules.items():
                    hidden_pretrained = module(hidden_pretrained)
                    if name == "layer1": break
            hidden_scratch = self.scratch_convnet_layer1(hidden_scratch)
            
            X_pretrained = torch.nn.functional.relu(self.oneXone_conv_layer1(hidden_pretrained))
            X_scratch = torch.cat([X_pretrained * self.pretrained_factor, hidden_scratch], dim=1)
            X_scratch = X_scratch + self.residual_conv_layer1(X_scratch)
            hidden_scratch = X_scratch
            
            # Layer 2
            with torch.no_grad():
                flag = False
                for name, module in self.pretrained_resnet._modules.items():
                    if flag:
                        hidden_pretrained = module(hidden_pretrained)
                        if name == "layer2": break
                    else:
                        if name == "layer1": flag = True
            hidden_scratch = self.scratch_convnet_layer2(hidden_scratch)
            
            X_pretrained = torch.nn.functional.relu(self.oneXone_conv_layer2(hidden_pretrained))
            X_scratch = torch.cat([X_pretrained * self.pretrained_factor, hidden_scratch], dim=1)
            X_scratch = X_scratch + self.residual_conv_layer2(X_scratch)
            hidden_scratch = X_scratch
            
            # hidden_scratch: n_camera * bsz, 128, 11, 11
            time_steps = 3
            X = hidden_scratch.view(-1, time_steps, hidden_scratch.shape[1], hidden_scratch.shape[2], hidden_scratch.shape[3])
            X_current = X[:, 1:, ...]
            X_previous = X_current - X[:, :time_steps - 1, ...].detach()
            X = torch.cat([X_current, X_previous], dim=1)
            X = X.view(bsz, -1, X.shape[3], X.shape[4])
            X = X.view(bsz, -1)
            
            return self.output_layer(X)


class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, feature_dim):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(self, h, state=None):
        state_en = self.state_encoder(state)
        h = h + state_en
        return self.fusion(h)
         

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, moe_gate_dim=256, moe_hidden_dim=256, num_experts=32, top_k=4, dropout=0.1, state_dim=None):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy1 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                     nn.ReLU(inplace=True))

        self.policy2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, action_shape[0]))
          
        self.moe = MoE( input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        gate_dim=moe_gate_dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        dropout=dropout,
                    )
        self.state_enc = StateEncoder(state_dim, hidden_dim, feature_dim)
            
        self.apply(utils.weight_init)

    def forward(self, obs, std, obs_sensor=None, metrics=None):
        h = self.trunk(obs)
        h = self.state_enc(h, obs_sensor)
        x = self.policy1(h)
        x, aux_loss = self.moe(x, metrics)
        
        mu = self.policy2(x)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, aux_loss


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, state_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.state_enc = StateEncoder(state_dim, hidden_dim, feature_dim)
        
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, obs_sensor=None):
        h = self.trunk(obs)
        h = self.state_enc(h, obs_sensor)
        
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class VNetwork(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim, state_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.V = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.state_enc = StateEncoder(state_dim, hidden_dim, feature_dim)

        self.apply(utils.weight_init)

    def forward(self, obs, obs_sensor=None):
        h = self.trunk(obs)
        h = self.state_enc(h, obs_sensor)
        v = self.V(h)
        return v


class MENTORAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, dormant_threshold,
                 target_dormant_ratio, dormant_temp, target_lambda,
                 lambda_temp, perturb_interval, min_perturb_factor,
                 max_perturb_factor, perturb_rate, num_expl_steps, stddev_type,
                 stddev_schedule, stddev_clip, expectile, use_tb,
                 lr_actor_ratio, aux_loss_scale_warmup, aux_loss_scale_warmsteps,
                 aux_loss_scale, aux_loss_type, encoder_type, resnet_fix,
                 oneXone_reg_scale, oneXone_reg_ratio, pretrained_factor, tp_set_size,
                 moe_gate_dim, moe_hidden_dim, num_experts, top_k, dropout, state_dim=None):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_type = stddev_type
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.dormant_threshold = dormant_threshold
        self.target_dormant_ratio = target_dormant_ratio
        self.dormant_temp = dormant_temp
        self.target_lambda = target_lambda
        self.lambda_temp = lambda_temp
        self.dormant_ratio = 1
        self.perturb_interval = perturb_interval
        self.min_perturb_factor = min_perturb_factor
        self.max_perturb_factor = max_perturb_factor
        self.perturb_rate = perturb_rate
        self.expectile = expectile
        self.awaken_step = None
        self.aux_loss_scale_warmup = aux_loss_scale_warmup
        self.aux_loss_scale_warmsteps = aux_loss_scale_warmsteps
        self.aux_loss_scale_max = aux_loss_scale
        self.aux_loss_scale = self.calc_aux_loss_scale(0)
        self.lr_actor_ratio = lr_actor_ratio
        self.aux_loss_type = aux_loss_type
        self.oneXone_reg_scale = oneXone_reg_scale
        self.oneXone_reg_ratio = oneXone_reg_ratio
        self.pretrained_factor = pretrained_factor
        self.pretrained_factor = pretrained_factor
        self.tp_set_size = tp_set_size
        self.moe_gate_dim = moe_gate_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        self.state_dim = state_dim

        # models
        self.encoder = Encoder(obs_shape, encoder_type, resnet_fix, pretrained_factor).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, 
                           moe_gate_dim, moe_hidden_dim, num_experts, top_k, dropout, state_dim).to(device)
        self.value_predictor = VNetwork(self.encoder.repr_dim, feature_dim, hidden_dim, state_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, state_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr * (1. if encoder_type == 'scratch' else 0.5))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr * self.lr_actor_ratio)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.predictor_opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.n_updates = 0
        self.perturb_time = 0
        self.train()
        self.critic_target.train()
        
        # Task-oriented Perturbation
        self.tp_set = utils.models_tuple(maxsize=self.tp_set_size, moe=True, gate=True)

    @property
    def dormant_stddev(self):
        return 1 / (1 + math.exp(-self.dormant_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def stddev(self, step):
        return self.dormant_stddev

    def perturb_factor(self):
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    @property
    def lambda_(self):
        return self.target_lambda / (1 + math.exp(self.lambda_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def calc_aux_loss_scale(self, step):
        if self.aux_loss_scale_warmup < 0 or self.aux_loss_scale_warmsteps < 0:
            return self.aux_loss_scale_max
        if step > self.aux_loss_scale_warmsteps:
            return self.aux_loss_scale_max
        return math.exp(
            math.log(self.aux_loss_scale_warmup) +  step / self.aux_loss_scale_warmsteps * ( math.log(self.aux_loss_scale_max) - math.log(self.aux_loss_scale_warmup) )
        )

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.value_predictor.train(training)

    def act(self, obs, step, eval_mode, obs_sensor=None):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))

        if obs_sensor is not None:
            obs_sensor = torch.as_tensor(obs_sensor, device=self.device)

        dist, _ = self.actor(obs, self.stddev(step), obs_sensor)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_predictor(self, obs, action, obs_sensor=None):
        metrics = dict()

        if obs_sensor is None:
            Q1, Q2 = self.critic(obs, action)
        else:
            Q1, Q2 = self.critic(obs, action, obs_sensor)
        Q = torch.min(Q1, Q2)
        if obs_sensor is None:
            V = self.value_predictor(obs)
        else:
            V = self.value_predictor(obs, obs_sensor)
        vf_err = V - Q
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 -
                                                                self.expectile)
        predictor_loss = (vf_weight * (vf_err**2)).mean()

        if self.use_tb:
            metrics['predictor_loss'] = predictor_loss.item()

        self.predictor_opt.zero_grad(set_to_none=True)
        predictor_loss.backward()
        self.predictor_opt.step()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step, obs_sensor=None, next_obs_sensor=None):
        metrics = dict()

        with torch.no_grad():
            if self.stddev_type == "drqv2":
                stddev = utils.schedule(self.stddev_schedule, step)
            elif self.stddev_type == "max":
                stddev = max(utils.schedule(self.stddev_schedule, step),
                             self.stddev(step))
            elif self.stddev_type == "dormant":
                stddev = self.stddev(step)
            elif self.stddev_type == "awake":
                if self.awaken_step == None:
                    stddev = self.stddev(step)
                else:
                    stddev = max(
                        self.stddev(step),
                        utils.schedule(self.stddev_schedule,
                                       step - self.awaken_step))
            else:
                raise NotImplementedError(self.stddev_type)
            if next_obs_sensor is None:
                dist, _ = self.actor(next_obs, stddev)
            else:
                dist, _ = self.actor(next_obs, stddev, next_obs_sensor)
            next_action = dist.sample(clip=self.stddev_clip)
            if next_obs_sensor is None:
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            else:
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action, next_obs_sensor)
            target_V_explore = torch.min(target_Q1, target_Q2)
            if next_obs_sensor is None:
                target_V_exploit = self.value_predictor(next_obs)
            else:
                target_V_exploit = self.value_predictor(next_obs, next_obs_sensor)
            target_V = self.lambda_ * target_V_exploit + (
                1 - self.lambda_) * target_V_explore
            target_Q = reward + (discount * target_V)

        if obs_sensor is None:
            Q1, Q2 = self.critic(obs, action)
        else:
            Q1, Q2 = self.critic(obs, action, obs_sensor)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        
        if self.oneXone_reg_scale > 0.01:
            def customized_regularization(weight):
                regularization = torch.norm(weight, p=2)
                return regularization
            critic_loss += self.oneXone_reg_scale * customized_regularization(self.encoder.oneXone_conv_layer1.weight, self.oneXone_reg_ratio)
            critic_loss += self.oneXone_reg_scale * customized_regularization(self.encoder.oneXone_conv_layer2.weight, self.oneXone_reg_ratio)
            
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, obs_sensor):
        metrics = dict()
        if self.stddev_type == "drqv2":
            stddev = utils.schedule(self.stddev_schedule, step)
        elif self.stddev_type == "max":
            stddev = max(utils.schedule(self.stddev_schedule, step),
                         self.stddev(step))
        elif self.stddev_type == "dormant":
            stddev = self.stddev(step)
        elif self.stddev_type == "awake":
            if self.awaken_step == None:
                stddev = self.stddev(step)
            else:
                stddev = max(
                    self.stddev(step),
                    utils.schedule(self.stddev_schedule,
                                   step - self.awaken_step))
        else:
            raise NotImplementedError(self.stddev_type)
        if obs_sensor is None:
            dist, aux_loss = self.actor(obs, stddev)
        else:
            dist, aux_loss = self.actor(obs, stddev, obs_sensor)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        if obs_sensor is None:
            Q1, Q2 = self.critic(obs, action)
        else:
            Q1, Q2 = self.critic(obs, action, obs_sensor)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        (actor_loss + aux_loss).backward()
        grad_mean = torch.mean(torch.abs(torch.cat([p.grad.flatten() for p in self.actor.parameters() if p.grad is not None])))
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.001)
        self.actor_opt.step()

        if self.use_tb:
            metrics['aux_loss'] = aux_loss.item()
            metrics['actor_grad_mean'] = grad_mean.item()
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def perturb(self):
        utils.perturb(self.actor, self.actor_opt, self.perturb_factor(), tp_set=self.tp_set, name="actor_enc")
        utils.perturb(self.actor.moe.experts, self.actor_opt, self.perturb_factor(), tp_set=self.tp_set, name="actor_moe_expert")
        utils.perturb(self.actor.moe.gate, self.actor_opt, self.perturb_factor(), tp_set=self.tp_set, name="actor_moe_gate")
        utils.perturb(self.critic, self.critic_opt, self.perturb_factor(), tp_set=self.tp_set, name="critic")
        utils.perturb(self.critic_target, self.critic_opt, self.perturb_factor(), tp_set=self.tp_set, name="critic_target")
        utils.perturb(self.value_predictor, self.predictor_opt, self.perturb_factor(), tp_set=self.tp_set, name="value_predictor")
        utils.perturb(self.encoder, self.encoder_opt, self.perturb_factor())

    def update(self, replay_iter, step):
        metrics = dict()

        self.n_updates += 1

        # aux_loss_scale
        self.aux_loss_scale = self.calc_aux_loss_scale(self.n_updates)

        # perturb
        if self.perturb_interval > 0 and self.n_updates % self.perturb_interval == 0:
            self.perturb()
            self.perturb_time += 1

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, obs_sensor, next_obs_sensor = utils.to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # calculate dormant ratio
        self.dormant_ratio, metrics = utils.cal_dormant_ratio(self.actor, obs.detach(), 0, obs_sensor,\
            percentage=self.dormant_threshold, metrics=metrics)

        if self.awaken_step is None and step > self.num_expl_steps and self.dormant_ratio < self.target_dormant_ratio:
            self.awaken_step = step

        if self.use_tb:
            metrics['perturb_time'] = self.perturb_time
            metrics['batch_reward'] = reward.mean().item()
            metrics['actor_dormant_ratio'] = self.dormant_ratio
            metrics['aux_loss_scale'] = self.aux_loss_scale
        
        # update predictor
        metrics.update(self.update_predictor(obs.detach(), action, obs_sensor))

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step, obs_sensor, next_obs_sensor))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step, obs_sensor))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
