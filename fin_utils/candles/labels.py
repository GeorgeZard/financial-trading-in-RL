import pandas as pd
import numpy as np
from typing import Dict
import warnings


def warn_commission_depr(commission, target_profit):
    if commission is not None and commission > 0.:
        warnings.warn("Keyword `commission` is deprecated use `target profit`")
        assert target_profit == 0, "`commission` and `target_profit` do the same thing please use `target_profit`"
        target_profit = commission
    return target_profit


def oracle_labels(candles: pd.DataFrame, target_profit: float = 0., include_exit: bool = False,
                  **kwargs):
    """
    Oracle labels based on the next open price

    :param candles:
    :param commission:
    :param include_exit: When true a zero label is used for when the commission is not worth changing position.
                         When false the the label of the previous position is propagated forward.
    :return:
    """

    target_profit = warn_commission_depr(kwargs.get('commission', None), target_profit)
    intra_candle_ret = (candles.open.shift(-1).ffill() / candles.open) - 1
    long = intra_candle_ret > target_profit
    short = intra_candle_ret < -target_profit
    positions = pd.Series(np.empty(intra_candle_ret.shape[0]), index=intra_candle_ret.index)
    positions.iloc[:] = 0 if include_exit else pd.NA
    positions[long] = 1
    positions[short] = -1
    positions = positions.shift(-1).ffill().fillna(0)
    return positions


def smooth_oracle_labels(candles: pd.DataFrame, target_profit: float = 0.,
                         avg_future_window: int = 10,
                         include_exit: bool = False, **kwargs):
    """
    Oracle labels based on the moving average of the next several prices to
    determine whether the correct position label

    :param candles:
    :param target_profit:
    :param commission:
    :param avg_future_window:
    :param include_exit: When true a zero label is used for when the commission is not worth changing position.
                         When false the the label of the previous position is propagated forward.
    :return:
    """
    target_profit = warn_commission_depr(kwargs.get('commission', None), target_profit)
    candles = candles[['open', 'high', 'low', 'close']]
    reverse_ma = candles.iloc[::-1].rolling(window=avg_future_window, min_periods=1).mean().iloc[::-1].mean(axis=1)
    intra_candle_ret = (reverse_ma / candles.open) - 1
    long = intra_candle_ret > target_profit
    short = intra_candle_ret < -target_profit
    positions = pd.Series(np.empty(intra_candle_ret.shape[0]), index=intra_candle_ret.index)
    positions.iloc[:] = 0 if include_exit else pd.NA
    positions[long] = 1
    positions[short] = -1
    positions = positions.shift(-1).ffill().fillna(0)
    return positions


def triple_barrier_labels(candles: pd.DataFrame, target_profit: float, barrier_length: int, use_hl: bool = False):
    """
    The triple barrier labels described in https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html

    :param candles: The price candles to use for the calculation
    :param target_profit: The percentage distance from the current price to the up and down barriers.
    :param barrier_length: The length of the barrier in the x axis.
    :param use_hl: whether to check the high and low prices instead of only the close price.
    :return:
    """
    candles: pd.DataFrame = candles[['open', 'high', 'low', 'close']]
    high = candles.high if use_hl else candles.close
    low = candles.low if use_hl else candles.close

    rolling_high = high.iloc[::-1].rolling(window=barrier_length).max().iloc[::-1]
    rolling_low = low.iloc[::-1].rolling(window=barrier_length).min().iloc[::-1]

    labels = pd.Series(np.zeros(candles.shape[0]), index=candles.index, dtype=np.int32)

    rolling_max_up = (rolling_high - candles.close) / candles.close
    rolling_max_down = (candles.close - rolling_low) / candles.close
    labels[rolling_max_up > target_profit] = 1
    labels[rolling_max_down > target_profit] = -1

    collision = (rolling_max_up > target_profit) & (rolling_max_down > target_profit)
    collision_idxs = candles.index.searchsorted(candles.index[collision])
    for idx in collision_idxs:
        cur_close = candles.close.iloc[idx]
        up_target = cur_close * (1 + target_profit)
        down_target = cur_close * (1 - target_profit)
        first_up_idx = (high.iloc[idx:idx + barrier_length] > up_target).argmax()
        first_dow_idx = (low.iloc[idx:idx + barrier_length] < down_target).argmax()
        if first_dow_idx < first_up_idx:
            labels.iloc[idx] = -1
        else:
            labels.iloc[idx] = 1

    return labels


def generate_optimal_limits(candles_dict: Dict[str, pd.DataFrame], on='prev_close',
                            clip=None, normalize='standardize', **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Generate optimal limit values for candles and normalize them by dividing with standard diviation.

    :param candles_dict: dictionary mapping pair name to candles
    :param clip: clip the limits. default to 2. (=200%)
    """
    optimals_dict = dict()
    for key in candles_dict.keys():
        c = candles_dict[key]
        if on == 'prev_close':
            low_limit = 1 - (c.low.shift(-1).ffill() / c.close)
            high_limit = (c.high.shift(-1).ffill() / c.close) - 1
        elif on == 'open':
            low_limit = (1 - (c.low / c.open)).shift(-1).ffill()
            high_limit = ((c.high / c.open) - 1).shift(-1).ffill()
        df = pd.concat([low_limit, high_limit], axis=1).astype(np.float32)
        if clip:
            df = df.clip(-clip, clip)
        optimals_dict[key] = df
    if normalize == 'standardize':
        std = pd.concat(optimals_dict.values(), axis=0).std()
        for key in candles_dict.keys():
            optimals_dict[key] = optimals_dict[key] / std
    elif normalize == 'qminmax':
        flat_vals = pd.concat(optimals_dict.values(), axis=0).values.flatten()
        quantiles = kwargs.get('quantile', [0.01, 0.95])
        min_limit = np.quantile(flat_vals, quantiles[0])
        max_limit = np.quantile(flat_vals, quantiles[1])
        for key in candles_dict.keys():
            optimals_dict[key] = ((optimals_dict[key] - min_limit) / (max_limit - min_limit)).clip(0.001, 0.999)
    return optimals_dict


def generate_prc_volatility_pred(candles_dict, average_window=100):
    """
    Generate labels of future volatility measured as an average of the percentage change between candles
    open to low and high to open.

    :param candles_dict:
    :param average_window:
    :return:
    """
    vol_dict = dict()
    for key, vals in candles_dict.items():
        c_vols = (vals.open / vals.low + vals.high / vals.open) * 0.5 - 1.
        c_vols = c_vols.iloc[::-1].rolling(window=average_window, min_periods=1).mean().iloc[::-1].ffill()
        vol_dict[key] = c_vols.to_frame('vols')
        assert all(c_vols.index == vals.index)
    std = pd.concat(vol_dict.values(), axis=0).std()
    mult = 1. / std
    for key, vols in vol_dict.items():
        vol_dict[key] = vols * mult
    return dict(vol_dict=vol_dict, std=std)


LABEL_GENERATORS = dict(
    oracle_labels=oracle_labels,
    smooth_oracle_labels=smooth_oracle_labels,
    triple_barrier_labels=triple_barrier_labels,
    generate_optimal_limits=generate_optimal_limits,
    generate_prc_volatility_pred=generate_prc_volatility_pred
)


def construct_labels(candle_dict: Dict[str, pd.DataFrame], **label_config):
    labels = dict()
    # backwards compatibility
    if 'label_config' in label_config:
        assert len(label_config) == 1
        label_config = label_config['label_config']
    # ------------------------
    for k, v in candle_dict.items():
        cur_labels = LABEL_GENERATORS[label_config['name']](v, **label_config.get('parameters', {}))
        # Set the shorting label "-1" as label 2 for one-hot encoding later.
        cur_labels[cur_labels == -1] = 2
        labels[k] = cur_labels.astype('int64').bfill()
    return labels
