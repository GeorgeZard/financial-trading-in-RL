from tsrl.utils.torch_base import TorchExperiment
from tsrl.torch_utils import to_np
from tsrl.torch_utils.optim import Lookahead, RAdam
from torch.optim import Adam
from tsrl.utils import create_decay, create_hyperbolic_decay, random_batched_iterator, RunningMeanStd, RunningScalarMean
from tsrl.environments.market_env import VecCandleMarketEnv
from tsrl.environments.wrappers import PytorchConverter, NormalizeWrapper
from tsrl.environments import generate_candle_features, create_pair_sample_ranges
from tsrl.advantage import NormalizedGAE, calculate_gae_hyperbolic, calculate_advantage_vectorized
from tsrl.algorithms.ppo import ppo_categorical_policy_loss, ppo_value_loss
from tsrl.experiments.market.model import MarketAgent
import random
from tsrl.environments import generate_candle_features


from fin_utils.pnl import pnl_from_price_position

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import kl_divergence, Categorical
from ray import tune

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import seaborn as sns

from plotly import graph_objects as go
import os
import warnings
import argparse
from torch.autograd import Variable
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



@torch.no_grad()
def gather_trajectories(env, model: MarketAgent, include_value=False, keep_states=False):
    """

    :param env: Environment to step through
    :param model: Model to use for inference through environment
    :param include_value: whether to include the value function estimation in the trajectory data
    :param keep_states: whether to keep the hidden state return by the model in the trajectory data (e.g. LSTM state)
    :return:
    """
    trajectories = defaultdict(list)
    rews, obs, is_first = env.observe()
    trajectories['obs'].append(obs)
    state = None
    while True:
        pred = model.sample_eval(state=state, **obs)
        if keep_states:
            trajectories['states'].append(state)
        state = pred['state']
        env.act(dict(market_action=pred['action']))
        rews, obs, is_first = env.observe()
        trajectories['action'].append(pred['action'].detach())
        trajectories['log_prob'].append(pred['log_prob'].detach())
        trajectories['logits'].append(pred['logits'].detach())
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
    action_tensor = torch.cat(trajectories['action'], dim=1).detach()
    logits = torch.cat(trajectories['logits'], dim=1).detach()
    if 'value' in trajectories:
        value_tensor = to_np(torch.stack(trajectories['value'], dim=1))
        return rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, logits, value_tensor
    return rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, logits


"""
Class for deep reinforcement learning enviroment -> Agent
"""

class MarketExperiment5(TorchExperiment):

    def __init__(self, *args, **kwargs):
        super(MarketExperiment5, self).__init__(*args, **kwargs)
        self.model: Optional[MarketAgent] = None
        self.teacher_model: Optional[MarketAgent] = None
        self.teacher_ensemble = {}
        self.optimizer = None

    def eval(self, data, from_checkpoint: Optional[int] = None,
             show_progress=True, batch_size=3000, warmup_window=10,
             from_date=None, to_date=None, pairs=None, dir=None):

        if self.model is None or isinstance(from_checkpoint, int):
            checkpoints = filter(lambda p: 'checkpoint' in p.stem and p.is_dir(), self.exp_path.iterdir())

            # last_checkpoint = sorted(checkpoints, key=lambda p: int(p.stem.split("_")[1]))[-1 or from_checkpoint]
            # exp_state_dict = torch.load(str(last_checkpoint / 'exp_state_dict.pkl'), map_location=self.device)
            exp_state_dict = torch.load(dir , map_location=self.device)
            model = MarketAgent(**self.db['model_params'])
            model.load_state_dict(exp_state_dict['model_state_dict'])
            model.to(self.device)
            model.eval()
            self.model = model
        else:
            model = self.model
        pairs = pairs or list(data['asset_index'].keys())
        asset_index_dict = {k: data['asset_index'][k] for k in pairs}
        idxs_ranges = create_pair_sample_ranges(asset_index_dict, freq='3M',
                                                from_date=from_date, to_date=to_date)

        vecenv = VecCandleMarketEnv(auto_reset=False, **data, **self.db['env_params'])
        env = PytorchConverter(vecenv, device=self.device)
        n_loops = int(np.ceil(len(idxs_ranges) / batch_size))
        pbar = tqdm(
            total=n_loops * max(idx['steps'] for idx in idxs_ranges),
            desc=f'Running Test Batch 1/{n_loops}. Batch Size {batch_size}',
            disable=not show_progress)
        info_list = defaultdict(list)
        last_reset = 0
        max_ep_len = self.db['env_params']['max_episode_steps']
        with torch.no_grad():
            for i in range(0, n_loops):
                batch_idxs = idxs_ranges[i * batch_size: (i + 1) * batch_size]
                start_idxs = np.array([v['start'] for v in batch_idxs])
                stop_idxs = np.array([v['stop'] for v in batch_idxs])
                vecenv.reset(stop_idxs=stop_idxs,
                             start_idxs=start_idxs,
                             pairs=[v['pair'] for v in batch_idxs])
                state = None
                rews, obs, is_first = env.observe()
                out = model(state=state, **obs)
                state = out['state']
                pobs = deque(maxlen=warmup_window)
                while True:
                    env.act(dict(market_action=out['market_action']))
                    rews, obs, is_first = env.observe()
                    for k, v in vecenv.get_info().items():
                        info_list[k].append(v)
                    pobs.append(obs)
                    last_reset += 1
                    if np.any(is_first):
                        if np.all(is_first):
                            break
                        vecenv.drop_envs(is_first)
                        for pi, obs_ in enumerate(pobs):
                            if any(is_first):
                                for k in obs.keys():
                                    obs_[k] = obs_[k][~is_first]
                        pbar.desc = f'Running Test Batch {i + 1}/{n_loops}. Batch Size {vecenv.num}'
                        for state_key, state_value in state.items():
                            state[state_key] = [st[:, ~is_first] for st in state_value]
                    if last_reset > max_ep_len:
                        for k, v in info_list.items():
                            info_list[k] = [np.concatenate(v)]
                        last_reset = 0
                        state = None
                        for pi, obs_ in enumerate(pobs):
                            pobs[pi] = obs_
                            out = model(state=state, **obs_)
                            state = out['state']
                    else:
                        out = model(state=state, **obs)
                        state = out['state']
                        pbar.update(1)

        pbar.close()
        info_list = {k: np.concatenate(v) for k, v in info_list.items()}
        df = pd.DataFrame.from_dict(info_list)
        del info_list
        res_dict = dict()
        for pair_encoding, df in df.groupby('pair_encoding'):
            df = df.drop('pair_encoding', axis=1)
            df.set_index('time_idx', inplace=True)
            df = df.iloc[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            pair = vecenv.pairs[pair_encoding]
            res_dict[pair] = df.iloc[:-1]
        return res_dict

    """
    :param data: training data with columns trade price, transaction price, position (-1, 0, 1), reward
    :param res_dict: evaluation data (same as data parameter)
    :train_end: end date training i.e 2021

    :return: a dictionary with keys for each asset i.e Bitcoin. Eth etc. For each asset we get the training pnl and the
     testing pnl with the excact date  and the pnl for this date i.e
     'BTCUSDT': {'train_pnl':
        2017-08-17 21:00:00    0.0,
        2017-08-17 22:00:00    0.0 
            ...
            ...
            }
            ...       
    """

    def detailed_backtest(self, data, res_dict, train_end):
        candle_dict, asset_index = data['candle_dict'], data['asset_index']
        pnl_ranges = dict()
        train_end = pd.to_datetime(train_end)

        for k in res_dict.keys():
            assert np.allclose(candle_dict[k].shape[0], res_dict[k].shape[0], atol=10, rtol=0)
            candles, trade_price, positions = candle_dict[k], res_dict[k]['trade_price'], res_dict[k]['position']
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close'], index=data['asset_index'][k])
            pnl = pnl_from_price_position(candles, trade_price=trade_price,
                                          positions=positions,
                                          commission=self.db['env_params']['commission_punishment'])
            train_end_idx = pnl.index.searchsorted(train_end)
            pnl_ranges[k] = dict(train_pnl=pnl.iloc[:train_end_idx],
                                 test_pnl=pnl.iloc[train_end_idx:])
        return pnl_ranges

    def backtest(self, data, res_dict, train_end, commission=None):
        candle_dict, asset_index = data['candle_dict'], data['asset_index']
        sum_ranges = dict()
        train_end = pd.to_datetime(train_end)
        for k in res_dict.keys():
            assert np.allclose(candle_dict[k].shape[0], res_dict[k].shape[0], atol=10, rtol=0)
            candles, trade_price, positions = candle_dict[k], res_dict[k]['trade_price'], res_dict[k]['position']
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close'], index=data['asset_index'][k])
            if commission is None:
                pnl = pnl_from_price_position(candles, trade_price=trade_price,
                                              positions=positions,
                                              commission=self.db['env_params']['commission_punishment'])
            else:
                pnl = pnl_from_price_position(candles, trade_price=trade_price,
                                              positions=positions,
                                              commission=commission)
            train_end_idx = pnl.index.searchsorted(train_end)
            sum_ranges[k] = dict(train_pnl=pnl.iloc[:train_end_idx].sum(),
                                 test_pnl=pnl.iloc[train_end_idx:].sum())
        global_sum_range = dict(train_pnl=sum([sr['train_pnl'] for sr in sum_ranges.values()]),
                                test_pnl=sum([sr['test_pnl'] for sr in sum_ranges.values()]))

        print("Back Testing")
        print(global_sum_range)
        return global_sum_range



    """
    Create different seeds -> for our model
    """
    def _fix_random_seed(self, manual_seed):
        # Fix seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        np.random.seed(manual_seed)
        torch.backends.cudnn.deterministic = True

        return manual_seed

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def teacher_train(self, data, teacher_model_params=None, env_params=None, n_epochs=100, batch_size=32, ppo_clip=0.2,
                      entropy_weight=0.01, train_range=("2000", "2016"), show_progress=True, gamma=0.9, tau=0.95,
                      value_horizon=np.inf, lr=5e-4, validation_interval=100, n_envs=128, n_reuse_policy=3,
                      n_reuse_value=1,
                      n_reuse_aux=0, weight_decay=0., checkpoint_dir=None, advantage_type='hyperbolic', use_amp=False,
                      env_step_init=1.0, rew_limit=6., recompute_values=False,
                      truncate_bptt=(5, 20), lookahead=False, seed=1, n_teachers = 1):
        # if 'model_params' in self.db and model_params is not None:
        #     self.logger.warning("model_params is already set. New Values ignored.")
        # self.db['model_params'] = model_params = self.db.get('model_params', model_params)
        #
        # if 'env_params' in self.db and env_params is not None:
        #     self.logger.warning("env_params is already set. New Values ignored.")
        # self.db['env_params'] = env_params = self.db.get('env_params', env_params)

        # Define the model -> Trading Agent
        # Teacher Model
        # print(teacher_model_params)
        # if self.teacher_model is None:
        #     num_inputs = list(data['feature_dict'].values())[0].shape[1]
        #     teacher_model_params['num_inputs'] = num_inputs
        #     # self.db['model_params'] = teacher_model_params
        #
        #     manual_seed = self._fix_random_seed(2)
        #     model1 = MarketAgent(**teacher_model_params)
        #
        #     manual_seed = self._fix_random_seed(3)
        #     model2 = MarketAgent(**teacher_model_params)
        #
        #     manual_seed = self._fix_random_seed(4)
        #     model3 = MarketAgent(**teacher_model_params)
        #
        #     manual_seed = self._fix_random_seed(5)
        #     model4 = MarketAgent(**teacher_model_params)
        #
        #     manual_seed = self._fix_random_seed(6)
        #     model5 = MarketAgent(**teacher_model_params)
        #
        #
        #     # self.teacher_ensemble = {'teacher 1': model1, 'teacher 2': model2}
        #
        #     # self.teacher_model = model1
        #
        #     self.teacher_ensemble = {'teacher 1': model1, 'teacher 2': model2, 'teacher 3': model3,
        #                             'teacher 4': model4, 'teacher 5': model5}
        teachers = []
        for i in range(7):
            teachers.append('teacher ' + str(i + 1))
        # Define the model -> Trading Agent
        if self.teacher_model is None:
            # Student model
            num_inputs = list(data['feature_dict'].values())[0].shape[1]
            teacher_model_params['num_inputs'] = num_inputs
            # self.db['teacher_model_params'] = teacher_model_params

            # Teacher Models
            manual_seed = self._fix_random_seed(seed + 2)
            model1 = MarketAgent(**teacher_model_params)

            manual_seed = self._fix_random_seed(seed + 3)
            model2 = MarketAgent(**teacher_model_params)

            manual_seed = self._fix_random_seed(seed + 4)
            model3 = MarketAgent(**teacher_model_params)

            manual_seed = self._fix_random_seed(seed + 5)
            model4 = MarketAgent(**teacher_model_params)

            manual_seed = self._fix_random_seed(seed + 6)
            model5 = MarketAgent(**teacher_model_params)

            manual_seed = self._fix_random_seed(seed + 7)
            model6 = MarketAgent(**teacher_model_params)

            manual_seed = self._fix_random_seed(seed + 8)
            model7 = MarketAgent(**teacher_model_params)

        model_dict = {'teacher 1': model1, 'teacher 2': model2, 'teacher 3': model3,
                      'teacher 4': model4, 'teacher 5': model5, 'teacher 6': model6,
                      'teacher 7': model7}

        # hold only the n_teacher from above model
        for t in range(n_teachers, len(model_dict)):
            del model_dict[teachers[t]]

        print(model_dict.keys())

        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = RAdam(model1.parameters(), lr=lr, weight_decay=weight_decay)
        if lookahead:
            optimizer = Lookahead(optimizer)
        self.optimizer = optimizer

        aux_optimizer = RAdam(model1.parameters(), lr=lr, weight_decay=weight_decay)
        # if lookahead:
        #     aux_optimizer = Lookahead(aux_optimizer)

        # lr_decay_lambda = create_hyperbolic_decay(
        #     1.0, 0.1, int(n_epochs * 0.8), hyperbolic_steps=10
        # )

        lr_decay_lambda = create_hyperbolic_decay(
            1, 0.1, int(n_epochs * 0.8), hyperbolic_steps=10
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
        aux_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(aux_optimizer, lr_lambda=lr_decay_lambda)

        return_stats = RunningMeanStd()
        cur_train_step = 0
        if checkpoint_dir:
            checkpoint_dict = self.load_checkpoint(checkpoint_dir, MarketAgent, load_rng_states=False)
            lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
            cur_train_step = checkpoint_dict.get('cur_train_step', 0)
            return_stats = checkpoint_dict.get('return_stats', RunningMeanStd())

        # summary_writer = SummaryWriter(str(self.exp_path), max_queue=500)
        env = VecCandleMarketEnv(
            n_envs=n_envs,
            sample_range=train_range,
            **env_params,
            **data)
        env = PytorchConverter(env, device=self.device)
        env = NormalizeWrapper(env, clip_reward=rew_limit)

        gae_engine = NormalizedGAE(advantage_type=advantage_type,
                                   return_stats=return_stats,
                                   device=self.device)

        for model in model_dict.items():
            model[1].to(self.device)

            training_loss = []
            # Training loop
            for gi in tqdm(range(cur_train_step, n_epochs), smoothing=0., disable=not show_progress):
                trajectories = gather_trajectories(env, model[1], include_value=not recompute_values)

                rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, old_logits, *value_tensor = \
                    prepare_trajectory_windows(trajectories)
                value_tensor = None if not value_tensor else value_tensor[0].squeeze()

                metrics_dict = dict(
                    rew_std=np.sqrt(env.rew_stats.var),
                    mean_reward=rew_array.sum(axis=1).mean(),
                    mean_action=action_tensor.float().mean().detach().cpu(),
                )
                if value_tensor is not None:
                    metrics_dict['mean_value'] = value_tensor.mean()

                # if (gi % 50) == 0:
                #     summary_writer.add_histogram('train/action_dist', action_tensor.detach(), gi, bins=5)
                #     summary_writer.add_histogram('train/value_dist', value_tensor.flatten(), gi, max_bins=60)

                model[1].train()

                # Policy Train
                avg_train_metrics = defaultdict(RunningScalarMean)
                hist_train_metrics = defaultdict(list)
                advantage_array, returns_array = gae_engine.advantage(
                    rew_array, value_tensor, gamma=gamma, tau=tau, horizon=value_horizon, clip=1.)

                for ti in range(n_reuse_policy):
                    # load these parameters in teacher model too, for distillation
                    for bi, (actions, old_log_prob, features, position, rews, old_logits_b, value_pred, advantage,
                             returns) in enumerate(
                        random_batched_iterator(
                            action_tensor, old_log_prob_tensor, obs_tensor_dict['features'],
                            obs_tensor_dict['position'], rew_array, old_logits, value_tensor,
                            advantage_array, returns_array,
                            batch_size=batch_size,
                        )):
                        optimizer.zero_grad(set_to_none=True)
                        train_eval = model[1].train_eval(actions=actions, features=features,
                                                      position=position)
                        # value_pred = to_np(train_eval['value'].squeeze()).astype(np.float32)
                        # advantage, returns = gae_engine.advantage(rews, value_pred)
                        policy_loss, pstats = ppo_categorical_policy_loss(actions, old_log_prob,
                                                                          Categorical(logits=old_logits_b),
                                                                          train_eval["pd"], advantage, ppo_clip=ppo_clip)

                        if entropy_weight > 0:
                            entropy_loss = -entropy_weight * train_eval["pd"].entropy().mean()
                            policy_loss += entropy_loss
                            avg_train_metrics['entropy_loss'].update(to_np(entropy_loss))

                        value_loss = torch.pow(train_eval['value'].squeeze(2) - returns, 2)[:, :-10].mean()
                        loss = policy_loss + value_loss

                        avg_train_metrics['value_loss'].update(to_np(value_loss))
                        avg_train_metrics['policy_loss'].update(to_np(policy_loss))
                        hist_train_metrics['advantage'].append(advantage.detach().flatten())
                        loss.backward()
                        nn.utils.clip_grad_norm_(model[1].parameters(), 1.)
                        optimizer.step()
                if n_reuse_aux > 0:
                    with torch.no_grad():
                        new_train_eval = model[1].train_eval(obs_tensor_dict['features'], obs_tensor_dict['position'],
                                                          action_tensor)
                        new_train_eval['logits'].detach_()
                    if lookahead:
                        optimizer.slow_step()
                # Aux Train
                for ti in range(n_reuse_aux):
                    for bi, (actions, features, position, rews, new_logits, value_pred, advantage, returns) in enumerate(
                            random_batched_iterator(
                                action_tensor, obs_tensor_dict['features'], obs_tensor_dict['position'],
                                rew_array, new_train_eval['logits'], value_tensor,
                                advantage_array, returns_array,
                                batch_size=batch_size,
                            )):
                        aux_optimizer.zero_grad(set_to_none=True)
                        train_eval = model1.train_eval(actions=actions, features=features,
                                                      position=position, truncate_bptt=truncate_bptt,
                                                      policy_aux_val=True)
                        # advantage, returns = gae_engine.advantage(rews, value_pred)
                        value_loss = torch.pow(train_eval['aux_val_pred'].squeeze(2) - returns, 2).mean()
                        clone_loss = kl_divergence(Categorical(logits=new_logits), train_eval['pd']).mean()
                        loss: torch.Tensor = value_loss + 20. * clone_loss
                        avg_train_metrics['aux_val_loss'].update(to_np(value_loss))
                        avg_train_metrics['aux_clone_loss'].update(to_np(clone_loss))
                        loss.backward()
                        nn.utils.clip_grad_norm_(model1.parameters(), 1.)
                        aux_optimizer.step()
                aux_lr_scheduler.step()
                lr_scheduler.step()
                avg_train_metrics = {k: v.mean for k, v in avg_train_metrics.items()}

                # training_loss.append(loss.item())
                # if ((gi + 1) % validation_interval) == 0:
                #     self.checkpoint(global_step=gi, return_stats=return_stats,
                #                     lr_scheduler_state_dict=lr_scheduler.state_dict(),
                #                     cur_train_step=gi)
                #     eval_dict = self.eval(data, show_progress=show_progress)
                #     cur_performance = self.backtest(data, eval_dict, train_end=train_range[1])
                #     cur_performance_pnl = self.backtest(data, eval_dict, train_end=train_range[1], commission=1e-3)

                    # if tune.session._session is None:
                    #     for k, v in cur_performance.items():
                    #         summary_writer.add_scalar(f'eval/{k}', v, gi)
                    #     for k, v in cur_performance_pnl.items():
                    #         summary_writer.add_scalar(f'pnl_/{k}', v, gi)
                    #     for k, v in metrics_dict.items():
                    #         summary_writer.add_scalar(f'metrics/{k}', v, gi)
                    #     for k, v in model.weight_stats().items():
                    #         summary_writer.add_scalar(f'model/{k}', v, gi)
                    #     for k, v in avg_train_metrics.items():
                    #         summary_writer.add_scalar(f'train/{k}', v, gi)
                    #     summary_writer.add_scalar(f'train/lr', lr_scheduler.get_last_lr()[0], gi)
                    #     # summary_writer.add_scalar(f'trades/train_trades', num_trades_train, gi)
                    #     # summary_writer.add_scalar(f'trades/test_trades', num_trades_test, gi)
                    #     # summary_writer.add_scalar(f'trades/train_exit', num_exit_train, gi)
                    #     # summary_writer.add_scalar(f'trades/test_exit', num_exit_test, gi)
                    # else:
                    #     tune.report(global_step=gi, lr=lr_scheduler.get_last_lr()[0],
                    #                 **cur_performance, **metrics_dict, **model.weight_stats(),
                    #                 **avg_train_metrics
                    #                 )

                    # del eval_dict

        # return train_eval

    def visualize_results(self, data, title):
        # for i, res in enumerate(data):
        #     plt.plot(res, '-x', label=legend[i])
        sns.set_theme()
        plt.plot(data, '-x')
        # plt.xticks(np.arange(0, epoch, 1))
        # plt.plot(data)
        plt.xlabel('No. of epoch')
        # plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def student_train(self, data, model_params=None, env_params=None, n_epochs=100, batch_size=32, ppo_clip=0.2,
                      entropy_weight=0.01, train_range=("2000", "2016"), show_progress=True, gamma=0.9, tau=0.95,
                      value_horizon=np.inf, lr=5e-4, validation_interval=100, n_envs=128, n_reuse_policy=3,
                      n_reuse_value=1,
                      n_reuse_aux=0, weight_decay=0., checkpoint_dir=None, advantage_type='hyperbolic', use_amp=False,
                      env_step_init=1.0, rew_limit=6., recompute_values=False,
                      truncate_bptt=(5, 20), lookahead=False, teacher_output=None, seed=0):
        #
        # if 'model_params' in self.db and model_params is not None:
        #     self.logger.warning("model_params is already set. New Values ignored.")
        # self.db['model_params'] = model_params = self.db.get('model_params', model_params)
        #
        if 'env_params' in self.db and env_params is not None:
            self.logger.warning("env_params is already set. New Values ignored.")
        self.db['env_params'] = env_params = self.db.get('env_params', env_params)

        print(model_params)
        # Define the model -> Trading Agent
        self.model = None
        if self.model is None:
            num_inputs = list(data['feature_dict'].values())[0].shape[1]
            model_params['num_inputs'] = num_inputs
            self.db['model_params'] = model_params
            manual_seed = self._fix_random_seed(1 + seed)

            model = MarketAgent(**model_params)
            model.to(self.device)
            self.model = model

        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if lookahead:
            optimizer = Lookahead(optimizer)
        self.optimizer = optimizer

        aux_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # if lookahead:
        #     aux_optimizer = Lookahead(aux_optimizer)

        # lr_decay_lambda = create_hyperbolic_decay(
        #     1.0, 0.1, int(n_epochs * 0.8), hyperbolic_steps=10
        # )

        lr_decay_lambda = create_hyperbolic_decay(
            1, 0.1, int(n_epochs * 0.8), hyperbolic_steps=10
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
        aux_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(aux_optimizer, lr_lambda=lr_decay_lambda)

        return_stats = RunningMeanStd()
        cur_train_step = 0
        if checkpoint_dir:
            checkpoint_dict = self.load_checkpoint(checkpoint_dir, MarketAgent, load_rng_states=False)
            lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
            cur_train_step = checkpoint_dict.get('cur_train_step', 0)
            return_stats = checkpoint_dict.get('return_stats', RunningMeanStd())

        summary_writer = SummaryWriter(str(self.exp_path), max_queue=500)
        env = VecCandleMarketEnv(
            n_envs=n_envs,
            sample_range=train_range,
            **env_params,
            **data)
        env = PytorchConverter(env, device=self.device)
        env = NormalizeWrapper(env, clip_reward=rew_limit)

        gae_engine = NormalizedGAE(advantage_type=advantage_type,
                                   return_stats=return_stats,
                                   device=self.device)
        cross_entropy = MineCrossEntropy(dim=1)
        # soft_max = nn.Softmax(dim=-1)
        training_loss = []

        # print(self.teacher_ensemble)
        #
        # print(self.model)

        # Training loop
        for gi in tqdm(range(cur_train_step, n_epochs), smoothing=0., disable=not show_progress):
            trajectories = gather_trajectories(env, model, include_value=not recompute_values)

            rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, old_logits, *value_tensor = \
                prepare_trajectory_windows(trajectories)
            value_tensor = None if not value_tensor else value_tensor[0].squeeze()

            metrics_dict = dict(
                rew_std=np.sqrt(env.rew_stats.var),
                mean_reward=rew_array.sum(axis=1).mean(),
                mean_action=action_tensor.float().mean().detach().cpu(),
            )
            if value_tensor is not None:
                metrics_dict['mean_value'] = value_tensor.mean()

            if (gi % 50) == 0:
                summary_writer.add_histogram('train/action_dist', action_tensor.detach(), gi, bins=5)
                summary_writer.add_histogram('train/value_dist', value_tensor.flatten(), gi, max_bins=60)

            model.train()

            # Policy Train
            avg_train_metrics = defaultdict(RunningScalarMean)
            hist_train_metrics = defaultdict(list)
            advantage_array, returns_array = gae_engine.advantage(
                rew_array, value_tensor, gamma=gamma, tau=tau, horizon=value_horizon, clip=1.)

            for ti in range(n_reuse_policy):
                # load these parameters in teacher model too, for distillation
                for bi, (actions, old_log_prob, features, position, rews, old_logits_b, value_pred, advantage,
                         returns) in enumerate(
                    random_batched_iterator(
                        action_tensor, old_log_prob_tensor, obs_tensor_dict['features'],
                        obs_tensor_dict['position'], rew_array, old_logits, value_tensor,
                        advantage_array, returns_array,
                        batch_size=batch_size,
                    )):
                    optimizer.zero_grad(set_to_none=True)
                    train_eval = model.train_eval(actions=actions, features=features,
                                                  position=position)
                    student_features = model.get_features(actions=actions, features=features,
                                                          position=position)
                    # value_pred = to_np(train_eval['value'].squeeze()).astype(np.float32)
                    # advantage, returns = gae_engine.advantage(rews, value_pred)
                    policy_loss, pstats = ppo_categorical_policy_loss(actions, old_log_prob,
                                                                      Categorical(logits=old_logits_b),
                                                                      train_eval["pd"], advantage, ppo_clip=ppo_clip)

                    if entropy_weight > 0:
                        entropy_loss = -entropy_weight * train_eval["pd"].entropy().mean()
                        policy_loss += entropy_loss
                        avg_train_metrics['entropy_loss'].update(to_np(entropy_loss))

                    value_loss = torch.pow(train_eval['value'].squeeze(2) - returns, 2)[:, :-10].mean()

                    pkt_losses = []
                    # load the values from ensemble teachers for current actions
                    # and take the mean of the output
                    for name, models in self.teacher_ensemble.items():
                        with torch.no_grad():
                            teacher_features = models.get_features(actions=actions, features=features,
                                                                   position=position)

                            # q(a|s) = res['logits'] -> teacher output distribution
                            # p(a|s) = train_eval['logits'] -> student output distribution
                            # cross_entropy(
                            pkt_losses.append(
                                cosine_similarity_loss(student_features.reshape(batch_size, 40 * 42),
                                                       teacher_features.reshape(batch_size, 40 * 42)))

                    # average of ensemble
                    tensor1 = torch.tensor(pkt_losses, requires_grad=True)
                    pkt_loss = torch.mean(tensor1)
                    # print(pkt_loss)

                    loss = policy_loss + value_loss + pkt_loss

                    avg_train_metrics['value_loss'].update(to_np(value_loss))
                    avg_train_metrics['policy_loss'].update(to_np(policy_loss))
                    hist_train_metrics['advantage'].append(advantage.detach().flatten())
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            if n_reuse_aux > 0:
                with torch.no_grad():
                    new_train_eval = model.train_eval(obs_tensor_dict['features'], obs_tensor_dict['position'],
                                                      action_tensor)
                    new_train_eval['logits'].detach_()
                if lookahead:
                    optimizer.slow_step()
            # Aux Train
            for ti in range(n_reuse_aux):
                for bi, (actions, features, position, rews, new_logits, value_pred, advantage, returns) in enumerate(
                        random_batched_iterator(
                            action_tensor, obs_tensor_dict['features'], obs_tensor_dict['position'],
                            rew_array, new_train_eval['logits'], value_tensor,
                            advantage_array, returns_array,
                            batch_size=batch_size,
                        )):
                    aux_optimizer.zero_grad(set_to_none=True)
                    train_eval = model.train_eval(actions=actions, features=features,
                                                  position=position, truncate_bptt=truncate_bptt,
                                                  policy_aux_val=True)
                    # advantage, returns = gae_engine.advantage(rews, value_pred)
                    value_loss = torch.pow(train_eval['aux_val_pred'].squeeze(2) - returns, 2).mean()
                    clone_loss = kl_divergence(Categorical(logits=new_logits), train_eval['pd']).mean()
                    loss: torch.Tensor = value_loss + 20. * clone_loss
                    avg_train_metrics['aux_val_loss'].update(to_np(value_loss))
                    avg_train_metrics['aux_clone_loss'].update(to_np(clone_loss))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    aux_optimizer.step()
            aux_lr_scheduler.step()
            lr_scheduler.step()
            avg_train_metrics = {k: v.mean for k, v in avg_train_metrics.items()}

            if ((gi + 1) % validation_interval) == 0 or gi==n_epochs-1:
                self.checkpoint(global_step=gi, return_stats=return_stats,
                                lr_scheduler_state_dict=lr_scheduler.state_dict(),
                                cur_train_step=gi)
                eval_dict = self.eval(data, show_progress=show_progress)
                cur_performance = self.backtest(data, eval_dict, train_end=train_range[1])
                # cur_performance_pnl = self.backtest(data, eval_dict, train_end=train_range[1], commission=1e-3)
                cur_performance_pnl = self.backtest(data, eval_dict, train_end=train_range[1], commission=2e-5)


                # num_trades_test = 1
                # num_trades_train = 1
                # num_exit_test = 0
                # num_pos_test = 0
                # num_exit_train = 0
                # num_pos_train = 0
                # for k in eval_dict.keys():
                #     train = eval_dict[k][eval_dict[k].index < pd.to_datetime(train_range[1])].position.diff()
                #     test = eval_dict[k][eval_dict[k].index >= pd.to_datetime(train_range[1])].position.diff()
                #     num_trades_train += train[train != 0].shape[0]
                #     num_trades_test += test[test != 0].shape[0]
                #     exit_train = eval_dict[k][eval_dict[k].index < pd.to_datetime(train_range[1])].position
                #     num_pos_train += exit_train.shape[0]
                #     num_exit_train += exit_train[exit_train == 0].shape[0]
                #     exit_test = eval_dict[k][eval_dict[k].index >= pd.to_datetime(train_range[1])].position
                #     num_exit_test += exit_test[exit_test == 0].shape[0]
                #     num_pos_test += exit_test.shape[0]
                # num_trades_test /= len(eval_dict.keys())
                # num_trades_train /= len(eval_dict.keys())
                # num_exit_train /= num_pos_train
                # num_exit_test /= num_pos_test
                if tune.is_session_enabled() is not None:
                    for k, v in cur_performance.items():
                        summary_writer.add_scalar(f'eval/{k}', v, gi)
                    for k, v in cur_performance_pnl.items():
                        summary_writer.add_scalar(f'pnl_/{k}', v, gi)
                    for k, v in metrics_dict.items():
                        summary_writer.add_scalar(f'metrics/{k}', v, gi)
                    for k, v in model.weight_stats().items():
                        summary_writer.add_scalar(f'model/{k}', v, gi)
                    for k, v in avg_train_metrics.items():
                        summary_writer.add_scalar(f'train/{k}', v, gi)
                    summary_writer.add_scalar(f'train/lr', lr_scheduler.get_last_lr()[0], gi)
                    # summary_writer.add_scalar(f'trades/train_trades', num_trades_train, gi)
                    # summary_writer.add_scalar(f'trades/test_trades', num_trades_test, gi)
                    # summary_writer.add_scalar(f'trades/train_exit', num_exit_train, gi)
                    # summary_writer.add_scalar(f'trades/test_exit', num_exit_test, gi)
                else:
                    tune.report(global_step=gi, lr=lr_scheduler.get_last_lr()[0],
                                **cur_performance, **metrics_dict, **model.weight_stats(),
                                **avg_train_metrics
                                )
                # if ((gi + 1) % 5000) == 0:
                #     detailed_pnls = self.detailed_backtest(data, eval_dict, train_end=train_range[1])
                #     fig = go.FigureWidget()
                #     for pair, pnls in detailed_pnls.items():
                #         train_pnl = pnls['train_pnl'].cumsum()
                #         test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
                #         fig.add_scatter(x=train_pnl.index, y=train_pnl, legendgroup=pair, name=pair)
                #         fig.add_scatter(x=test_pnl.index, y=test_pnl, legendgroup=pair, name=pair)
                #     fig.write_html(f'plots/figure_{lr}_{gi+1}.html')
                #     del detailed_pnls
                del eval_dict
        return training_loss


"""
   Defining loss of the model 
   L_D  = - 1/N * Σ Σ (q(a_i|s) * log (p (a_i|s)) )

   L_D = cross_entropy(logits(student), logits(teacher))

   Final_loss = policy_loss + values_loss + cross_entropy(train_eval['logits'], teacher_train_eval['logits'])
   = L_RL + L_D
"""

class MineCrossEntropy(nn.Module):
    def __init__(self, dim=-1):
        super(MineCrossEntropy, self).__init__()
        self.dim = dim

    def forward(self, q, p):
        p = p.log_softmax(dim=self.dim)

        return torch.mean(torch.sum(-q * p, dim=self.dim))

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # print(output_net.shape)
    # print(target_net.shape)
    # print(output_net)
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss