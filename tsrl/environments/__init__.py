import pandas as pd
import numba as nb
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from fin_utils.candles.features import construct_features
from tsrl.utils.memoize import memoize


"""
Comment line 28 if you want only price infos for your data
"""
def resample_bars(period, symbol, path, base=None, field_to_use='trade',variable = False, start_time='09:00', end_time='17:00'):

    if field_to_use == 'trade':
        what_to_use = ''
    store = pd.HDFStore(f'{path}{symbol}_minutes{what_to_use}.hdf5', mode='r')
    bars = store[symbol]
    store.close()
    if "volume" in bars.columns:
        bars = bars.resample(period, label='right', closed='right', offset=base).agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last', 'volume':'sum', 'sentiment':'mean'})
    else:
        bars = bars.resample(period, label='right', closed='right', offset=base).agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last','sentiment':'mean'})

    bars['sentiment'] = bars['sentiment'].interpolate('linear')
    bars = bars.dropna()
    bars.drop(['sentiment'], axis=1, inplace=True)
    # print(str(bars.index[0]))
    # test = str(bars.index[0])
    # print(test[0:4])

    return bars

@nb.njit()
def nbsign(x):
    return nb.int8(-1 if x < 0 else (1 if x > 0 else 0))


def create_sample_range(idx: pd.DatetimeIndex, sample_range: Tuple[str, str]) -> Tuple[int, int]:
    start = idx.searchsorted(pd.to_datetime(sample_range[0]).to_datetime64().astype(idx.dtype))
    stop = idx.searchsorted(pd.to_datetime(sample_range[1]).to_datetime64().astype(idx.dtype))
    return int(start), int(stop)


def create_pair_sample_ranges(asset_index_dict: Dict[str, np.ndarray], freq='6M', from_date=None, to_date=None):
    idxs_ranges = []
    keys = asset_index_dict.keys()
    for i, k in enumerate(keys):
        asset_index = asset_index_dict[k]
        fdt = from_date or asset_index[0]
        tdt = to_date or asset_index[-1]
        dt_range = pd.date_range(fdt, tdt, freq=freq).to_numpy()
        idxs = np.searchsorted(asset_index, dt_range)
        max_idxs = len(asset_index) - 2
        for j in range(1, idxs.shape[0]):
            toidx = min(idxs[j], max_idxs)
            fromidxs = idxs[j - 1]
            if toidx - fromidxs <= 2:
                continue
            if asset_index[0] <= dt_range[j] and asset_index[-1] >= dt_range[j - 1]:
                idxs_ranges.append(dict(pair=k, start=fromidxs, stop=toidx, steps=toidx - fromidxs))

        # Ensure the edges of the date range are included
        if dt_range[0] > fdt:
            start_idx = 0 if fdt <= asset_index[0] else asset_index.searchsorted(fdt)
            idxs_ranges.append(dict(pair=k, start=start_idx, stop=idxs[0], steps=idxs[0] - start_idx))
        if dt_range[-1] < tdt:
            stop_idx = (max_idxs if tdt >= asset_index[-1] else min(asset_index.searchsorted(tdt), max_idxs))
            if stop_idx - idxs[-1] > 10:
                idxs_ranges.append(dict(pair=k, start=idxs[-1], stop=stop_idx, steps=stop_idx - idxs[-1]))
    return idxs_ranges


"""
Loading the data from folder -> need to change the path in 
feather_folder to your local Path
"""
def generate_candle_features(train_end, timescale: str, pairs: Optional[List[str]] = None,
                             keep_dfs=False,
                             feather_folder=Path('~/financial_data/histdata_feathers'),
                             feature_config=(
                                     dict(func_name='internal_bar_diff', columns=['close', 'high', 'low'],
                                          use_pct=True),
                                     dict(func_name='inter_bar_changes', use_pct=True),
                                     dict(func_name='hl_to_pclose'))):

    candle_dict = {}
    for pair in pairs:
        df = resample_bars(timescale, pair, feather_folder)
        """
        Check if training period is at least 2 years
        """
        starting_date = str(df.index[0])
        starting_year = int(starting_date[0:4])

        if starting_year <= 2019:
            candle_dict[pair] = df
            print(pair + " Accepted!")
    print(len(candle_dict))
    feature_dict = construct_features(candle_dict, train_end, feature_config=feature_config)
    asset_index = dict()
    print(feature_dict['BTCUSDT'].columns)
    for k in feature_dict.keys():
        idx: np.ndarray = np.array(feature_dict[k].index.to_pydatetime(), dtype=np.datetime64)
        assert (idx == candle_dict[k].index).all()
        assert (~pd.isna(feature_dict[k])).all().all()
        assert (~pd.isna(candle_dict[k])).all().all()
        asset_index[k] = idx
        assert np.all(np.diff(idx) > np.timedelta64(0))
    candle_df_dict = candle_dict.copy()
    if not keep_dfs:
        candle_dict = {k: np.ascontiguousarray(v[['open', 'high', 'low', 'close']].values.astype('float32')) for k, v in
                       candle_dict.items()}
        feature_dict = {k: np.ascontiguousarray(v.values.astype('float32')) for k, v in feature_dict.items()}

    # print(feature_dict['BTCUSDT'])
    return dict(asset_index=asset_index, candle_dict=candle_dict,
                feature_dict=feature_dict, candle_df_dict=candle_df_dict), len(candle_dict)


DEFAULT_V2_PAIRS = [
    'nzdusd', 'gbpusd', 'xauaud', 'gbpnzd', 'usdmxn', 'gbpchf', 'audusd', 'usdjpy',
    'gbpaud', 'usdcad', 'eurcad', 'xauchf', 'eurnzd', 'eurczk', 'usddkk', 'usdzar', 'audnzd',
    'usdsgd', 'xauusd', 'audjpy', 'usdpln', 'gbpcad', 'audchf', 'eurgbp', 'usdnok', 'eurjpy', 'eursek',
    'eurtry', 'nzdcad', 'nzdchf', 'audcad', 'cadchf', 'xaugbp', 'eurhuf', 'bcousd', 'xagusd',
    'zarjpy', 'usdtry', 'eurpln', 'nzdjpy', 'usdczk', 'sgdjpy', 'chfjpy', 'usdhuf',
    'xaueur', 'eurchf', 'eurusd', 'euraud', 'cadjpy', 'usdsek', 'usdchf', 'gbpjpy',

    # pegged
    # 'eurdkk', 'usdhkd',
    # slow
    # 'eurnok',

    # oil
    # 'wtiusd',

    # indices
    # 'hkxhkd', 'auxaud','frxeur',
    # 'jpxjpy', 'spxusd', 'udxusd',
    # 'ukxgbp', 'nsxusd', 'grxeur', 'etxeur',
]
# print(len(DEFAULT_V2_PAIRS))
