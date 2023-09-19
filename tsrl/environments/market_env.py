import gym3
import pandas as pd
import numpy as np
from gym3.types import TensorType, DictType, Real, Discrete
from typing import Dict, Tuple, Any, List, Union, Optional
from tsrl.utils import fast_categorical
from tsrl.environments import create_sample_range, nbsign

import random
import numba as nb
from numba.core import types


def dict_to_nbdict(x: Dict[str, np.ndarray], value_type=None):
    """
    Convert a python dictionary to a numba dictionary. The resulting dictionary is slow when used outside
    a numba function.

    :param x:
    :param value_type:
    :return:
    """
    if value_type is None:
        value_type = nb.typeof(x[list(x.keys())[0]])
    d = nb.typed.Dict.empty(
        key_type=types.string,
        value_type=value_type,
    )
    d.update(**x)
    return d


@nb.njit
def vec_observe(current_market_positions, feature_array, cur_pairs, cur_idxs, candle_array, last_act):
    """
    Extract observation from multiple environments

    :param current_market_positions:
    :param feature_array:
    :param cur_pairs:
    :param cur_idxs:
    :param candle_array:
    :return:
    """
    positions = nb.typed.List()  # .empty_list(nb.float32[:])
    features = nb.typed.List()  # .empty_list(nb.float32[:])
    # opt_next_lim = nb.typed.List()
    for position, cur_idx, pair, lact in zip(current_market_positions, cur_idxs, cur_pairs, last_act):
        cur_feature_values = feature_array[pair]
        if position < 0:
            position = 2
        elif position > 0:
            position = 1
        positions.append(fast_categorical(position, num_classes=3))
        features.append(cur_feature_values[cur_idx - 1])
        # cur_candles = candle_array[pair]
        # h, l, c = cur_candles[cur_idx, 1:4]
        # prev_c = cur_candles[cur_idx - 1, 3]
        # opt_next_lim.append(abs(1 - (hl/prev_c)).min())
        # opt_next_lim_ = (h / prev_c) - 1
        # opt_next_lim_ = np.where((lact == 2) | (lact == 4), 1 - (l / prev_c), opt_next_lim_)
        # opt_next_lim.append(opt_next_lim_)
        # next_idx = min(cur_idx + 1, len(cur_candles))
        # nh, nl = cur_candles[next_idx, 1:3]
        #
        # if lact == 3:
        #     opt_next_lim.append(1 - 5e-5 - (l / prev_c))
        # elif lact == 4:
        #     opt_next_lim.append((h / prev_c) - (1 + 5e-5))
        # else:
        #     opt_next_lim.append(min((h / prev_c) - (1 + 5e-5), (1 - 5e-5) - (l / prev_c)))
    # observation = {'features': features, 'position': positions}
    return features, positions


@nb.generated_jit()
def vec_act(market_actions, limit_actions, current_market_positions, candle_array, cur_idxs, cur_pairs,
            last_transacted_prices, pos_avg_prices, commission_punishment, ensure_exec,
            last_trade_prices, limit_market_equalizer, precomputed_limits):
    """
    Act on multiple environments

    :param market_actions:
    :param limit_actions:
    :param current_market_positions:
    :param candle_array:
    :param cur_idxs:
    :param cur_pairs:
    :param last_transacted_prices:
    :param pos_avg_prices:
    :param commission_punishment:
    :param ensure_exec:
    :param last_trade_prices:
    :return:
    """

    if isinstance(precomputed_limits, types.NoneType):
        def vec_act_with_limits(
                market_actions, limit_actions, current_market_positions, candle_array, cur_idxs, cur_pairs,
                last_transacted_prices, pos_avg_prices, commission_punishment, ensure_exec,
                last_trade_prices, limit_market_equalizer, precomputed_limits):
            res_list = []
            for i in nb.prange(len(cur_pairs)):
                candles = candle_array[cur_pairs[i]]
                res_list.append(handle_action(market_action=market_actions[i], limit_action=limit_actions[i],
                                              current_market_position=current_market_positions[i],
                                              cur_candles=candles, cur_idx=cur_idxs[i],
                                              last_transacted_price=last_transacted_prices[i],
                                              pos_avg_price=pos_avg_prices[i],
                                              commission_punishment=commission_punishment,
                                              ensure_exec=ensure_exec, last_trade_price=last_trade_prices[i],
                                              limit_market_equalizer=limit_market_equalizer))
            # res_list = res_list
            reward = nb.typed.List([r[0] for r in res_list])
            last_transacted_price = nb.typed.List([r[1] for r in res_list])
            pos_avg_price = nb.typed.List([r[2] for r in res_list])
            last_trade_price = nb.typed.List([r[3] for r in res_list])
            current_market_position = nb.typed.List([r[4] for r in res_list])
            return reward, last_transacted_price, pos_avg_price, last_trade_price, current_market_position

        return vec_act_with_limits
    else:
        def vec_act_with_precomp_limits(
                market_actions, limit_actions, current_market_positions, candle_array, cur_idxs, cur_pairs,
                last_transacted_prices, pos_avg_prices, commission_punishment, ensure_exec,
                last_trade_prices, limit_market_equalizer, precomputed_limits):
            res_list = nb.typed.List()
            for i in nb.prange(len(cur_pairs)):
                candles = candle_array[cur_pairs[i]]
                res_list.append(handle_action(market_action=market_actions[i],
                                              limit_action=precomputed_limits[cur_pairs[i]][cur_idxs[i]],
                                              current_market_position=current_market_positions[i],
                                              cur_candles=candles, cur_idx=cur_idxs[i],
                                              last_transacted_price=last_transacted_prices[i],
                                              pos_avg_price=pos_avg_prices[i],
                                              commission_punishment=commission_punishment,
                                              ensure_exec=ensure_exec, last_trade_price=last_trade_prices[i],
                                              limit_market_equalizer=limit_market_equalizer))
            # res_list = res_list
            reward = nb.typed.List([r[0] for r in res_list])
            last_transacted_price = nb.typed.List([r[1] for r in res_list])
            pos_avg_price = nb.typed.List([r[2] for r in res_list])
            last_trade_price = nb.typed.List([r[3] for r in res_list])
            current_market_position = nb.typed.List([r[4] for r in res_list])
            return reward, last_transacted_price, pos_avg_price, last_trade_price, current_market_position

        return vec_act_with_precomp_limits


"""
Get the action
"""
@nb.njit
def handle_action(market_action, limit_action, current_market_position, cur_candles, cur_idx,
                  last_transacted_price, pos_avg_price, commission_punishment, ensure_exec,
                  last_trade_price, limit_market_equalizer):
    reward = 0
    o, h, l, c = cur_candles[cur_idx]
    next_o = cur_candles[cur_idx + 1, 0]
    # prev_c = cur_candles[cur_idx - 1, 3]
    prev_c = o
    trade_args = (current_market_position, last_transacted_price,
                  pos_avg_price, commission_punishment)
    tr_flag = False
    amount_to_positive = 1 - min(0, current_market_position)
    amount_to_negative = -1 - max(0, current_market_position)
    if market_action == 0 and current_market_position != 0:
        trade_res = trade_exec(o, -current_market_position, *trade_args)
        tr_flag = True
    elif market_action == 1 and current_market_position <= 0:
        trade_res = trade_exec(o, amount_to_positive, *trade_args)
        tr_flag = True
    elif market_action == 2 and current_market_position >= 0:
        trade_res = trade_exec(o, amount_to_negative, *trade_args)
        tr_flag = True
    elif market_action == 3 and current_market_position <= 0:
        buy_lo_price = (1. - limit_action[0]) * prev_c
        # buy_lo_price = max(buy_lo_price, l * 1.00001)
        buy_lo_price = min(buy_lo_price, o)
        # buy_lo_price = l * 1.00001
        if buy_lo_price >= l:
            trade_res = trade_exec(buy_lo_price, amount_to_positive, *trade_args)
            if o > c:
                reward -= limit_market_equalizer * (o - buy_lo_price) / pos_avg_price
            tr_flag = True
        elif ensure_exec:
            trade_res = trade_exec(c, amount_to_positive, *trade_args)
            tr_flag = True
    elif market_action == 4 and current_market_position >= 0:
        sell_lo_price = (1 + limit_action[1]) * prev_c
        # sell_lo_price = min(sell_lo_price, h * 0.99999)
        sell_lo_price = max(sell_lo_price, o)
        # sell_lo_price = h * 0.99999
        if sell_lo_price <= h:
            trade_res = trade_exec(sell_lo_price, amount_to_negative, *trade_args)
            if o < c:
                reward -= limit_market_equalizer * (sell_lo_price - o) / pos_avg_price
            tr_flag = True
        elif ensure_exec:
            trade_res = trade_exec(c, amount_to_negative, *trade_args)
            tr_flag = True
    if tr_flag:
        last_transacted_price = trade_res['last_transacted_price']
        pos_avg_price = trade_res['pos_avg_price']
        last_trade_price = trade_res['last_trade_price']
        reward += trade_res['reward']
        current_market_position = trade_res['current_market_position']

    obtained_pnl = (next_o - last_transacted_price) / pos_avg_price
    reward += obtained_pnl * current_market_position
    last_transacted_price = next_o
    return (reward,
            last_transacted_price,
            pos_avg_price,
            last_trade_price,
            current_market_position)


@nb.njit
def find_opt_limits(current_market_positions, cur_idxs, cur_pairs_enc, candle_array):
    opt_next_lim = []
    lim_eps = 1e-5
    for position, cur_idx, pair in zip(current_market_positions, cur_idxs, cur_pairs_enc):
        cur_candles = candle_array[pair]
        o, h, l, c = cur_candles[cur_idx]
        prev_c = o  # cur_candles[cur_idx - 1, 3]
        opt_next_lim.append((1 - lim_eps - (l / prev_c), (h / prev_c) - (1 + lim_eps)))
    return np.asarray(opt_next_lim)


"""
Trading action
"""
@nb.njit
def trade_exec(price, amount, current_market_position, last_transacted_price,
               pos_avg_price, commission_punishment):
    reward = nb.float32(0.0)
    cur_sign = nbsign(current_market_position)
    if current_market_position == 0:
        pos_avg_price = price
        current_market_position += amount
    elif cur_sign != nbsign(amount):
        reward += nb.float32(((price - last_transacted_price) / pos_avg_price) * current_market_position)
        current_market_position += amount
        if cur_sign != nbsign(current_market_position):
            pos_avg_price = price
    else:
        # print("Adding to existing pos")
        sum_pos = current_market_position + amount
        pos_avg_price = (current_market_position / sum_pos) * pos_avg_price + (
                amount / sum_pos
        ) * price
        current_market_position = sum_pos
    reward -= nb.float32(commission_punishment * abs(amount))
    last_trade_price = nb.float32(price)
    last_transacted_price = nb.float32(price)
    return {"reward": reward,
            "current_market_position": current_market_position,
            "last_transacted_price": last_transacted_price,
            "pos_avg_price": pos_avg_price,
            "last_trade_price": last_trade_price}


@nb.njit
def vec_info(asset_index_array, cur_pairs, cur_idxs):
    time_idxs_list = nb.typed.List()
    for i, (cpair, cidx) in enumerate(zip(cur_pairs, cur_idxs)):
        time_idxs_list.append(asset_index_array[cpair][cidx - 1])
    return time_idxs_list


import warnings


"""
Observations provide the location of the target and agent 
"""
class VecCandleMarketEnv(gym3.Env):
    def __init__(self,
                 feature_dict: Dict[str, np.ndarray],
                 candle_dict: Dict[str, np.ndarray],
                 asset_index: Dict[str, pd.DatetimeIndex],
                 max_episode_steps=1000,
                 sample_range=('2000', '2030'),
                 commission_punishment=2e-5,
                 ensure_exec=False, n_envs=1,
                 static_limit=0.,
                 auto_reset=True,
                 limit_market_equalizer=0.,
                 precomputed_limits=None,
                 **kwargs
                 ):
        # warnings.warn(f"Unused kwargs: {list(kwargs.keys())}")
        k_ = list(feature_dict.keys())[0]
        self.auto_reset = auto_reset
        self.limit_market_equalizer = limit_market_equalizer
        self.precomputed_limits = precomputed_limits
        ob_space = DictType(
            features=TensorType(Real(), shape=(feature_dict[k_].shape[1],)),
            position=TensorType(Real(), shape=(1,)),
            # opt_next_lim=TensorType(Real(), shape=(1,)),
        )
        ac_space = DictType(
            market_action=TensorType(Discrete(5, dtype_name='int32'), shape=(1,)),
            limit_action=TensorType(Real(), shape=(1,)),
        )
        self.commission_punishment = np.float32(commission_punishment)
        self.sample_range = sample_range
        self.remaining_steps = None
        self.pairs = list(candle_dict.keys())
        self.pair_encoding = np.arange(len(self.pairs), dtype=np.int32)
        self.feature_array = nb.typed.List([feature_dict[self.pairs[i]] for i in self.pair_encoding])
        self.candle_array = nb.typed.List([candle_dict[self.pairs[i]] for i in self.pair_encoding])
        self.asset_index_array = nb.typed.List([asset_index[self.pairs[i]] for i in self.pair_encoding])
        if self.precomputed_limits is not None:
            self.precomputed_limits = nb.typed.List(
                [self.precomputed_limits[self.pairs[i]].values for i in self.pair_encoding])
        self.max_episode_steps = max_episode_steps
        self.rewards = np.zeros(n_envs, dtype=np.float32)
        self.ensure_exec = ensure_exec
        self.static_limit = np.float32(static_limit)

        # Create sample ranges for each pair
        self.sample_range_dict = []
        ignore_pairs = []
        for i, asset_idx in enumerate(self.asset_index_array):
            cur_sample_range = create_sample_range(asset_idx, sample_range)
            self.sample_range_dict.append(cur_sample_range)
            if cur_sample_range[1] - cur_sample_range[0] < max_episode_steps:
                ignore_pairs.append(i)
        # Remove pairs that have too few samples available in the sample_range period.
        np.delete(self.pair_encoding, ignore_pairs)

        super(VecCandleMarketEnv, self).__init__(ob_space=ob_space, ac_space=ac_space, num=n_envs)
        self.reset()

    def set_max_episode_steps(self, steps):
        self.max_episode_steps = steps

    def set_limit_market_equalizer(self, limit_market_equalizer):
        self.limit_market_equalizer = limit_market_equalizer

    def reset(self, pairs: Optional[List[Union[int, str]]] = None, start=None, stop=None, start_idxs=None,
              stop_idxs=None):
        if pairs is None:
            pairs = np.random.choice(self.pair_encoding, self.num)
        elif isinstance(pairs[0], str):
            pairs = np.asarray([self.pairs.index(pair) for pair in pairs])
        if len(pairs) != self.num:
            self.num = len(pairs)
        self.current_market_positions = nb.typed.List(np.zeros(self.num, dtype=np.float32))
        if start_idxs is not None:
            self.cur_idxs = start_idxs
            self.remaining_steps = stop_idxs - start_idxs
        else:
            self.cur_idxs = np.empty(self.num, dtype=np.int64)
            stop_idxs = []
            for i, pair in enumerate(pairs):
                index = self.asset_index_array[pair]
                if start:
                    cur_idx = index.searchsorted(pd.to_datetime(start))
                else:
                    sample_range = self.sample_range_dict[pair]
                    cur_idx = random.randint(sample_range[0], sample_range[1] - self.max_episode_steps - 1)

                if stop:
                    stop_idx = self.asset_index_array[pair].searchsorted(pd.to_datetime(stop))
                    stop_idx = min(stop_idx, self.asset_index_array[pair].shape[0] - 1)
                    self.remaining_steps = stop_idx - cur_idx
                    assert stop_idx > cur_idx
                else:
                    stop_idx = cur_idx + self.max_episode_steps
                stop_idxs.append(stop_idx)
                self.cur_idxs[i] = cur_idx
        self.remaining_steps = np.array(stop_idxs) - self.cur_idxs
        self.cur_pairs_enc = pairs
        self.is_first = np.ones(self.num).astype('bool')
        self.last_transacted_prices = np.array(
            [self.candle_array[k][self.cur_idxs[i], 3] for i, k in enumerate(pairs)])
        self.last_trade_prices = nb.typed.List(self.last_transacted_prices.copy())
        self.last_transacted_prices = nb.typed.List(self.last_transacted_prices)
        self.pos_avg_prices = self.last_transacted_prices
        self.last_act = np.zeros(self.num)
        self.rewards = np.zeros(self.num, dtype=np.float32)

    def drop_envs(self, idxs):
        if any(idxs):
            idxs = np.argwhere(~idxs)[:, 0]
            self.current_market_positions = nb.typed.List([self.current_market_positions[idx] for idx in idxs])
            self.cur_idxs = self.cur_idxs[idxs]
            self.cur_pairs_enc = nb.typed.List([self.cur_pairs_enc[idx] for idx in idxs])
            self.last_transacted_prices = nb.typed.List([self.last_transacted_prices[idx] for idx in idxs])
            self.last_trade_prices = nb.typed.List([self.last_trade_prices[idx] for idx in idxs])
            self.rewards = nb.typed.List([self.rewards[idx] for idx in idxs])
            self.pos_avg_prices = nb.typed.List([self.pos_avg_prices[idx] for idx in idxs])
            self.is_first = self.is_first[idxs]
            self.remaining_steps = self.remaining_steps[idxs]
            self.num = idxs.shape[0]

    def set_commission(self, commission):
        self.commission_punishment = commission[0]
        return (None,)

    def observe(self) -> Tuple[Any, Any, Any]:
        features, positions = vec_observe(self.current_market_positions, self.feature_array,
                                          self.cur_pairs_enc,
                                          self.cur_idxs, self.candle_array,
                                          self.last_act)
        observation = {'features': features, 'position': positions}
        return np.array(self.rewards), observation, self.is_first

    def find_opt_limits(self):
        return find_opt_limits(self.current_market_positions, self.cur_idxs, self.cur_pairs_enc,
                               self.candle_array)

    def get_info(self):
        time_idxs = vec_info(self.asset_index_array, self.cur_pairs_enc, self.cur_idxs)
        return dict(
            time_idx=np.asarray(time_idxs),
            pair_encoding=self.cur_pairs_enc.copy(),
            trade_price=np.asarray(self.last_trade_prices, dtype=np.float32),
            transaction_price=np.asarray(self.last_transacted_prices, dtype=np.float32),
            position=np.asarray(self.current_market_positions, dtype=np.float32),
            reward=np.asarray(self.rewards, dtype=np.float32))

    def act(self, ac: dict) -> None:
        self.is_first = np.repeat([False], self.num)
        market_actions = ac['market_action'][:, 0]
        if 'limit_action' not in ac:
            limit_actions = np.repeat(self.static_limit, self.num)[:, None]
            limit_actions[market_actions == 3] *= -1
            warnings.warn("limit action not in ac")
        else:
            limit_actions = ac['limit_action']
        self.last_act = market_actions
        results = vec_act(
            market_actions, limit_actions, self.current_market_positions,
            self.candle_array, self.cur_idxs, self.cur_pairs_enc,
            self.last_transacted_prices, self.pos_avg_prices, self.commission_punishment,
            self.ensure_exec, self.last_trade_prices, self.limit_market_equalizer, self.precomputed_limits)
        self.rewards, self.last_transacted_prices, \
        self.pos_avg_prices, self.last_trade_prices, \
        self.current_market_positions = results

        self.cur_idxs += 1
        self.remaining_steps -= 1
        self.is_first = self.remaining_steps <= 0

        if self.auto_reset and all(self.remaining_steps == 0):
            self.reset()


if __name__ == '__main__':
    """Example run testing the raw speed of the environment"""

    from tqdm import tqdm
    from tsrl.environments import generate_candle_features

    pairs = [
        "eurusd",
        "eurjpy", "eurgbp", "eurchf", "gbpchf", "audjpy", "audusd", "chfjpy", 'gbpjpy',
        'usdcad', 'audcad', 'euraud', 'cadjpy', 'gbpaud', 'usdjpy', 'gbpusd', 'gbpcad', 'cadchf',
        'usdchf', 'eurcad', "nzdchf", 'gbpnzd', 'eurnzd', "nzdusd", 'nzdjpy', 'audchf', 'nzdcad',
        'usdpln', 'usdzar', 'eurtry', 'usdhkd', 'usdczk', 'zarjpy',
        'usdhuf', 'usdnok', 'eursek', 'eurhuf', 'sgdjpy', 'usdsgd', 'eurnok',
        'eurczk', 'usddkk', 'eurpln', 'audnzd', 'usdtry', 'usdsek',
    ]
    data = generate_candle_features('2016', '4H', pairs=pairs,
                                    feature_config=(
                                        dict(name='int_bar_changes', func_name='inter_bar_changes',
                                             columns=['close', 'high', 'low'],
                                             use_pct=True),
                                        dict(name='int_bar_changes_50', func_name='inter_bar_changes',
                                             columns=['close', 'high', 'low'], use_pct=True,
                                             smoothing_window=50),
                                        dict(func_name='internal_bar_diff', use_pct=True),
                                        dict(func_name='hl_to_pclose'),
                                        dict(name='hlvol1', func_name='hl_volatilities', smoothing_window=10),
                                        dict(name='hlvol2', func_name='hl_volatilities', smoothing_window=50),
                                        dict(name='rvol1', func_name='return_volatilities', smoothing_window=10),
                                        dict(name='rvol2', func_name='return_volatilities', smoothing_window=50),
                                        dict(func_name='time_feature_day'),
                                        dict(func_name='time_feature_year'),
                                        dict(func_name='time_feature_month'),
                                        dict(func_name='time_feature_week')
                                    ))
    import pickle
    from pathlib import Path

    # with open(Path('~/rl_experiments/combined_limit_pretrained.pkl').expanduser(), 'rb') as fp:
    #     precomputed_limits = pickle.load(fp)
    # data['precomputed_limits'] = precomputed_limits
    env = VecCandleMarketEnv(n_envs=256, **data)
    rs, obs, first = env.observe()
    env.act(gym3.types_np.sample(env.ac_space, (env.num,)))
    for i in tqdm(range(1000000), smoothing=0.):
        rs, obs, first = env.observe()
        action = gym3.types_np.sample(env.ac_space, (env.num,))
        action.pop('limit_action')
        env.act(action)
        # env.get_info()
