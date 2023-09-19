from collections import defaultdict
from tsrl.model import BaseModel
import numpy as np
import torch


def gather_trajectories_statefull(env, model: BaseModel, include_value=False):
    """

    :param env: Environment to step through
    :param model: Model to use for inference through environment
    :return:
    """
    trajectories = defaultdict(list)
    with torch.no_grad():
        rews, obs, is_first = env.observe()
        trajectories['obs'].append(obs)
        while True:
            pred = model.stateful_step(**obs)
            env.act(dict(market_action=pred['action']))
            rews, obs, is_first = env.observe()
            trajectories['action'].append(pred['action'].detach())
            trajectories['log_prob'].append(pred['log_prob'].detach())
            if include_value:
                trajectories['value'].append(pred['value'].detach().to('cpu', non_blocking=True))
            trajectories['rews'].append(rews)
            if any(is_first):
                assert all(is_first)
                break
            trajectories['obs'].append(obs)
    return trajectories


def prepare_trajectory_windows(trajectories):
    # obs shapes : step x n_envs x nb_features : transpose(0,1)
    rew_array = np.asarray(trajectories['rews']).T
    obs_tensor_dict = dict()
    for k in trajectories['obs'][0].keys():
        obs_tensor_dict[k] = torch.stack([tob[k] for tob in trajectories['obs']], dim=1).detach()
    old_log_prob_tensor = torch.stack(trajectories['log_prob'], dim=1).detach()
    assert old_log_prob_tensor.shape[-1] == 1
    old_log_prob_tensor = old_log_prob_tensor[..., 0]
    action_tensor = torch.stack(trajectories['action'], dim=1).detach()
    # value_tensor = to_np(torch.stack(trajectories['value'], dim=1))
    return rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor
