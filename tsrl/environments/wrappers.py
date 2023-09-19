import torch
import numpy as np
import pandas as pd
from typing import Tuple, Any
from gym3.wrapper import Wrapper as gym3Wrapper
from gym3.types import DictType
from tsrl.torch_utils import to_np
from collections import defaultdict

from tsrl.utils import RunningMeanStd


class Wrapper(gym3Wrapper):
    def __getattr__(self, item):
        getattr(self.env, item)


class ActionRecorder(Wrapper):
    def __init__(self, *args, **kwargs):
        super(ActionRecorder, self).__init__(*args, **kwargs)
        self.record_dict = defaultdict(lambda: defaultdict(list))

    def act(self, ac: Any) -> None:
        info_list = self.get_info()
        for info in info_list:
            # info['ac'] = ac
            pair = info.pop('pair')
            for k, v in info.items():
                self.record_dict[pair][k].append(v)
        self.env.act(ac)

    def export_records(self):
        export_dict = dict()
        for k, v in self.record_dict.items():
            df = pd.DataFrame.from_dict(v)
            df.set_index('time_idx', inplace=True)
            df = df.iloc[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            export_dict[k] = df
        return export_dict


class PytorchConverter(Wrapper):

    def __init__(self, *args, device=torch.device('cpu'), **kwargs):
        super(PytorchConverter, self).__init__(*args, **kwargs)
        self.device = device
        self.obs_keys = None
        self.ac_keys = None
        assert type(self.ac_space) is DictType and type(self.ob_space) is DictType

    # @profile
    def observe(self) -> Tuple[Any, dict, Any]:
        rews, obs, is_first = self.env.observe()
        if self.obs_keys is None:
            self.obs_keys = list(obs.keys())
        for k in self.obs_keys:
            obs[k] = torch.as_tensor(np.asarray(obs[k]).astype(np.float32), dtype=torch.float32, device=self.device)
        return rews, obs, is_first

    # @profile
    def act(self, ac: dict) -> None:
        if self.ac_keys is None:
            self.ac_keys = list(ac.keys())
        for k in self.ac_keys:
            ac[k] = to_np(ac[k])
        return self.env.act(ac)


eps = 1e-8


class NormalizeWrapper(Wrapper):
    def __init__(self, *args, clip_reward=None, rew_stats=None, **kwargs):
        super(NormalizeWrapper, self).__init__(*args, **kwargs)
        self.rew_stats = rew_stats or RunningMeanStd()
        self.clip_reward = clip_reward

    # @profile
    def observe(self) -> Tuple[Any, Any, Any]:
        rews, obs, is_first = self.env.observe()
        self.rew_stats.update(rews.flatten())
        rews /= np.sqrt(self.rew_stats.var + eps)
        if self.clip_reward:
            rews = np.clip(rews, -self.clip_reward, self.clip_reward)
        return rews, obs, is_first


class LimitActionConverter(Wrapper):
    def __init__(self, *args, output_range, **kwargs):
        super(LimitActionConverter, self).__init__(*args, **kwargs)
        self.output_range = output_range
        self.last_market_action = None
        self.last_limit_action = None

    def act(self, ac: Any) -> None:
        # ac['limit_action'] = (ac['limit_action'] - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        ac['limit_action'] = (ac['limit_action'] * (self.output_range[1] - self.output_range[0])) + self.output_range[0]
        self.last_market_action = ac['market_action']
        self.last_limit_action = ac['limit_action']
        return super(LimitActionConverter, self).act(ac)

    def reverse_lims(self, lims):
        return (lims - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
