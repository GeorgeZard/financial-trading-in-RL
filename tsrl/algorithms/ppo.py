import torch
from torch.distributions import Categorical, kl_divergence

from typing import Optional, List

"""
Proximal Policy Optimization (ppo) loss compute it with this objective function:
                    L_clip(θ) = E_t[min(r_t(θ)A, r_t_clip(θ)A]
where r_t(θ) = (π_θ(α|s_t))/(π_θ_old(α|s_t))
      r_t_clip = clip(r_t(θ), 1-ε, 1+ε) 

By utilizing a clipped version of the ratio r_t(θ) around the value
of 1 within ϵ, the policy exploration can be constrained to the
close vicinity in the parameter space

:return : policy loss, stats
"""

def ppo_categorical_policy_loss(old_actions, old_log_prob, old_pd, dist: torch.distributions.Distribution,
                                advantage, ppo_clip=0.2, kl_beta=0., reverse_kl_beta=0.,
                                mask=None):
    # old_log_prob = Categorical(logits=old_logits).log_prob(old_actions)
    stats = dict()
    ratio = (dist.log_prob(old_actions) - old_log_prob).exp()
    objective = advantage * ratio
    if ppo_clip > 0:
        clipped_objective = advantage * ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
        objective = torch.min(objective, clipped_objective)
    if mask is not None:
        objective = objective[mask]
    objective = - objective.mean()
    stats['objective'] = objective
    loss: torch.Tensor = objective
    if kl_beta:
        kl_loss = kl_divergence(old_pd, dist)
        if mask is not None:
            kl_loss = kl_loss[mask]
        kl_loss = kl_loss.mean()
        stats['kl_loss'] = kl_loss
        loss += kl_loss
    if reverse_kl_beta > 0:
        kl_loss = kl_divergence(dist, old_pd)
        if mask is not None:
            kl_loss = kl_loss[mask]
        kl_loss = kl_loss.mean()
        stats['reverse_kl_loss'] = kl_loss
        loss += kl_loss
    return loss, stats


"""
Expected values of returns and values according with the above objective function.  
"""
def ppo_value_loss(returns, value):
    value_loss = torch.pow(value - returns, 2).mean()
    return value_loss


# def ppg_auxiliary(old_logits, dist: torch.distributions.Distribution, returns, value,
#                   beta_clone=1.):
#     value_loss = torch.pow(value - returns, 2).mean()
#     clone_loss = kl_divergence(Categorical(logits=old_logits), dist).mean()
#     loss: torch.Tensor = value_loss + beta_clone * clone_loss
#     return loss
