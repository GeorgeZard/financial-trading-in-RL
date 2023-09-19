import numba as nb
import numpy as np
import torch
from tsrl.utils import RunningMeanStd


class NormalizedGAE:
    def __init__(self, advantage_type=None, return_stats=None, device=torch.device('cpu')):
        self.type = advantage_type
        assert advantage_type in ['hyperbolic', 'exponential', 'direct_reward']
        self.return_stats = return_stats or RunningMeanStd()
        self.device = device

    def advantage(self, rews, value_pred, tau=0.95, horizon=np.inf,
                  gamma=0.95, clip=None, mult=None):
        value_pred = value_pred * max(np.sqrt(self.return_stats.var), 1e-6) + self.return_stats.mean
        if self.type == 'hyperbolic':
            advantage, returns = calculate_gae_hyperbolic(
                rews, value_pred,
                hb_discount=gamma, tau=tau, horizon=horizon)
        elif self.type == 'exponential':
            advantage, returns = calculate_advantage_vectorized(
                rews, value_pred,
                gamma=gamma, tau=tau, horizon=horizon)
        elif self.type == 'direct_reward':
            advantage, returns = rews.copy(), rews.copy()
        else:
            raise NotImplementedError()
        advantage = torch.as_tensor(advantage, device=self.device, dtype=torch.float32)
        self.return_stats.update(returns.flatten())
        # returns = (returns - self.return_stats.mean) / max(np.sqrt(self.return_stats.var), 1e-6)
        returns = returns / max(np.sqrt(self.return_stats.var), 1e-6)
        returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)
        # advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-6)
        advantage = advantage / max(advantage.std(), 1e-6)
        if mult:
            advantage *= mult
        if clip:
            advantage = advantage.clamp(-clip, clip)
        return advantage, returns


@nb.njit()
def calculate_advantage_vectorized(reward, value, gamma, tau, horizon=np.inf):
    assert reward.ndim == 2
    episode_len = reward.shape[1]
    advantage = np.zeros_like(reward)
    returns = np.zeros_like(reward)
    advantage[:, -1] = reward[:, -1] - value[:, -1]
    returns[:, -1] = reward[:, -1]
    steps = np.arange(episode_len - 1)[::-1]
    if np.isinf(horizon):
        for step in steps:
            delta = reward[:, step] + gamma * value[:, step + 1] - value[:, step]
            advantage[:, step] = delta + advantage[:, step + 1] * tau * gamma
    else:
        discount = (tau * gamma) ** np.arange(episode_len)
        discount[horizon:] = 0.
        discount = discount.reshape((1, -1))
        deltas = np.empty_like(reward)
        deltas[:, -1] = reward[:, -1] - value[:, -1]
        deltas[:, :-1] = reward[:, :-1] + gamma * value[:, 1:] - value[:, :-1]
        for step in steps:
            current_horizon = episode_len - (step + 1)
            advantage[:, step] = deltas[:, step] + (deltas[:, step + 1:] * discount[:, :current_horizon]).sum(axis=1)
    return advantage, advantage + value


@nb.njit(parallel=False)
def calculate_gae_hyperbolic(reward, value, hb_discount, tau, horizon=np.inf):
    assert reward.ndim == 2
    episode_len = reward.shape[1]
    final_advantages = np.zeros_like(reward)
    final_advantages[:, -1] = reward[:, -1] - value[:, -1]
    discounts = 1 / (1 + hb_discount * (np.arange(episode_len) + 1))
    discounts = discounts.reshape((1, -1))
    if np.isinf(horizon):
        horizon = episode_len
    for total in np.arange(episode_len):
        cur_rewards, cur_value = reward[:, -total - 1:], value[:, -total - 1:]
        cur_len = cur_rewards.shape[1]
        cur_steps = np.arange(cur_len - 1)
        advantage = np.zeros((reward.shape[0], cur_len), dtype=np.float32)
        advantage[:, -1] = final_advantages[:, -1]
        for step in cur_steps[:horizon][::-1]:
            cur_discount = discounts[0, step]
            delta = cur_rewards[:, step] + cur_discount * cur_value[:, step + 1] - cur_value[:, step]
            advantage[:, step] = delta + advantage[:, step + 1] * (cur_discount * tau)
        final_advantages[:, -total - 1] = advantage[:, 0]
    return final_advantages, final_advantages + value
