import random
import re
import time

from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from queue import PriorityQueue

import sys
sys.path.append(".")


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)


def cal_dormant_ratio(model, *inputs, percentage=0.025, seq=False, metrics=None):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)
    layer_id = 0
    for module, hook in zip(
        (module for module in model.modules() if isinstance(module, nn.Linear)), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                if seq:
                    mean_output = output_data.abs().mean((0, 1))
                else:
                    mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indices)  
                if metrics is not None:   
                    metrics['layer_{}_dormant_ratio'.format(layer_id)] = len(dormant_indices) / module.weight.shape[0]
                layer_id += 1

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons / total_neurons, metrics

def get_module_by_name(module, name):
    """
    Gets a submodule from a module by its string name.
    Args:
        module (nn.Module): The parent module.
        name (str): The dot-separated name of the submodule.
    Returns:
        nn.Module: The requested submodule.
    """
    parts = name.split('.')
    # Handle the case of the root module itself when name is ''
    if name == '':
        return module
    
    for part in parts:
        module = getattr(module, part)
    return module


def perturb(net, optimizer, perturb_factor, tp_set=None, name="actor"):
    if tp_set is None:
        linear_keys = [
            name for name, mod in net.named_modules()
            if isinstance(mod, torch.nn.Linear)
        ]
        new_net = deepcopy(net)
        new_net.apply(weight_init)

        for n, param in net.named_parameters():
            if any(key in n for key in linear_keys):
                noise = new_net.state_dict()[n] * (1 - perturb_factor)
                param.data = param.data * perturb_factor + noise
            else:
                param.data = net.state_dict()[n]
        optimizer.state = defaultdict(dict)
        return net, optimizer
    else:
        linear_keys = [
            name for name, mod in net.named_modules()
            if (isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d))
        ]
        new_net = deepcopy(net)
        new_net.apply(weight_init)
        if name == "actor_enc":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.actors)))
        elif name == "actor_moe_expert":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.moes, moe_expert=True)), moe_expert=True)
        elif name == "actor_moe_gate":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.gates, moe_gate=True)), moe_gate=True)
        elif name == "critic":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.critics)))
        elif name == "critic_target":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.critic_targets)))
        elif name == "value_predictor":
            tp_set.sampled_model(new_net, tp_set.sample_params(tp_set.cal_params_stats(tp_set.value_predictors)))

        for n, param in net.named_parameters():     
            if any(key in n for key in linear_keys):
                param.data = param.data * perturb_factor\
                    + new_net.state_dict()[n] * (1 - perturb_factor)
            else:
                param.data = net.state_dict()[n]
        optimizer.state = defaultdict(dict)
        return net, optimizer


class models_tuple(object):
    def __init__(self, maxsize=128, moe=False, gate=False):
        self.maxsize = maxsize
        self.length = 0
        self.episode_reward = []
        self.actors = []
        self.critics = []
        self.critic_targets = []
        self.value_predictors = []
        self.moe = moe
        if self.moe:
            self.moes = []
        self.gate = gate
        if self.gate:
            self.gates = []

    def add(self, episode_reward, actor, critic, critic_target, value_predictor, moe=None, gate=None):
        if self.length < self.maxsize:
            self.episode_reward.append(episode_reward)
            self.actors.append(actor)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.value_predictors.append(value_predictor)
            self.length += 1               
            if self.moe:
                self.moes.append(moe)
            if self.gate:
                self.gates.append(gate)
        else:
            min_idx = self.episode_reward.index(min(self.episode_reward))
            if episode_reward > self.episode_reward[min_idx]:
                self.episode_reward[min_idx] = episode_reward
                self.actors[min_idx] = actor
                self.critics[min_idx] = critic
                self.critic_targets[min_idx] = critic_target
                self.value_predictors[min_idx] = value_predictor
                if self.moe:
                    self.moes[min_idx] = moe
                if self.gate:
                    self.gates[min_idx] = gate
    
    def log(self, metrics):
        metrics['tp_set_mean_episode_reward'] = np.mean(self.episode_reward)
        return metrics

    # def cal_params_stats(self, models, moe_expert=False, moe_gate=False):
    #     weights_and_biases = []
    #     for name, param in models[0].named_modules():
    #         if not (moe_expert or moe_gate) and "moe" not in name and (isinstance(param, nn.Linear) or isinstance(param, nn.Conv2d)) and "." in name:
    #             layer_weights = torch.stack([model._modules[name.split(".")[0]][int(name.split(".")[1])].weight.data for model in models])
    #             layer_biases = torch.stack([model._modules[name.split(".")[0]][int(name.split(".")[1])].bias.data for model in models])
    #             weights_and_biases.append(layer_weights)
    #             weights_and_biases.append(layer_biases)
    #         elif (moe_expert or moe_gate) and isinstance(param, nn.Linear) and "." in name:
    #             layer_weights = torch.stack([model._modules[name.split(".")[0]][int(name.split(".")[1])].weight.data for model in models])
    #             layer_biases = torch.stack([model._modules[name.split(".")[0]][int(name.split(".")[1])].bias.data for model in models])
    #             weights_and_biases.append(layer_weights)
    #             weights_and_biases.append(layer_biases)
    #         elif not (moe_expert or moe_gate) and isinstance(param, nn.Linear):
    #             layer_weights = torch.stack([model.weight.data for model in models])
    #             weights_and_biases.append(layer_weights)
    #     params_stats = [(param.mean(dim=0), torch.clamp(param.std(dim=0), min=1e-7)) for param in weights_and_biases]
    #     return params_stats

    def cal_params_stats(self, models, moe_expert=False, moe_gate=False):
        # If the list of models is empty, there are no stats to calculate.
        if not models:
            return []

        params_stats = []
        # Use the first model in the list as a reference for architecture
        for layer_name, layer_module in models[0].named_modules():
            
            # --- Define conditions to select which layers to process ---
            is_linear_or_conv = isinstance(layer_module, (nn.Linear, nn.Conv2d))
            is_moe_layer = "moe" in layer_name

            # Skip layers that are not Linear or Conv2d
            if not is_linear_or_conv:
                continue
            
            # Logic to include/exclude layers based on moe_expert/moe_gate flags
            if moe_expert or moe_gate:
                # If we are in MoE mode, only process layers that are part of MoE
                if not is_moe_layer:
                    continue
            else:
                # If we are in the default mode (e.g., "actor_enc"), skip MoE layers
                if is_moe_layer:
                    continue

            # --- If the layer is selected, gather it from all models in the set ---
            try:
                # Use the helper to get the corresponding layer from each model in the list
                corresponding_layers = [get_module_by_name(m, layer_name) for m in models]
                
                # --- Stack weights and biases to calculate stats ---
                layer_weights = torch.stack([l.weight.data for l in corresponding_layers])
                params_stats.append((layer_weights.mean(dim=0), torch.clamp(layer_weights.std(dim=0), min=1e-7)))

                # Check if the layers have a bias attribute before trying to access it
                if corresponding_layers[0].bias is not None:
                    layer_biases = torch.stack([l.bias.data for l in corresponding_layers])
                    params_stats.append((layer_biases.mean(dim=0), torch.clamp(layer_biases.std(dim=0), min=1e-7)))

            except (AttributeError, RuntimeError) as e:
                print(f"Skipping layer '{layer_name}' due to an error: {e}")
                # This might happen if models in the set have slightly different architectures
                # or if a layer is missing a weight/bias.
                continue
                
        return params_stats

    def sample_params(self, params_stats):
        sampled_params = []
        for mean, std in params_stats:
            dist = pyd.Normal(mean, std)
            sampled_params.append(dist.sample())
        return sampled_params

    def sampled_model(self, model, sampled_params, moe_expert=False, moe_gate=False):
        # If there are no sampled parameters, there's nothing to do.
        if not sampled_params:
            return

        # A single index to track our position in the flat list of sampled parameters.
        param_idx = 0
        
        # Iterate through the modules of the model we want to update.
        for layer_name, layer_module in model.named_modules():

            # --- Define conditions to select which layers to process ---
            is_linear_or_conv = isinstance(layer_module, (nn.Linear, nn.Conv2d))
            is_moe_layer = "moe" in layer_name

            # Skip modules that are not Linear or Conv2d layers.
            if not is_linear_or_conv:
                continue
            
            # Logic to include/exclude layers based on moe_expert/moe_gate flags
            if moe_expert or moe_gate:
                if not is_moe_layer:
                    continue
            else: # Default case
                if is_moe_layer:
                    continue

            # --- If the layer is selected, assign its new parameters ---
            # Check if we still have enough parameters in our list.
            if param_idx >= len(sampled_params):
                # print(f"Warning: Ran out of sampled_params while trying to update layer '{layer_name}'.")
                break

            # Assign the weight
            layer_module.weight.data = sampled_params[param_idx]
            param_idx += 1

            # If the layer has a bias, assign it from the next spot in the list.
            if layer_module.bias is not None:
                if param_idx >= len(sampled_params):
                    # print(f"Warning: Ran out of sampled_params for bias of layer '{layer_name}'.")
                    break
                layer_module.bias.data = sampled_params[param_idx]
                param_idx += 1
