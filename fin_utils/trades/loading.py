from functools import partial
import pandas as pd
import numpy as np
from pathlib import Path


def group_ticks(ticks, time_group_span=None, price_group_span=None,
                time_col='date', price_col='price', side_col='side',
                size_col='quantity'):
    if time_group_span:
        ticks.loc[:, time_col] = ticks[time_col].dt.ceil(time_group_span)
    if price_group_span:
        ticks.loc[:, price_col] = ticks[price_col] - np.remainder(ticks[price_col].values, price_group_span)
    groupby = ticks.groupby([time_col, price_col, side_col])
    ticks = groupby[size_col].sum().reset_index()
    return ticks


def load_trade_feather(path: Path, time_group_span=None, price_group_span=None, index_col='date'):
    df = pd.read_feather(str(path))
    cols = df.columns.tolist()
    assert all([c in cols for c in ['price', 'quantity', 'date', 'side']])
    if index_col:
        df.set_index(index_col, inplace=True)
    if time_group_span or price_group_span:
        df = group_ticks(df, time_group_span=time_group_span, price_group_span=price_group_span)
    return path.stem.lower(), df


def load_trades_feather_dir(path: Path, pairs=None, index_col='date', n_workers=None):
    if type(pairs) is str:
        pairs = [pairs]
    path = Path(path).expanduser()
    feather_dict = dict()
    dirlist = list(path.iterdir())
    flist = [f for f in dirlist if f.suffix == '.feather']
    if len(flist) < len(dirlist):
        ignored_files = set(dirlist) - set(flist)
        ignored_files = [f.name for f in ignored_files]
        print(f"Ignoring non-feather files: {','.join(ignored_files)}")
    if pairs is not None:
        pairs = [p.lower() for p in pairs]
        flist = filter(lambda f: f.stem.lower() in pairs, flist)
    if not n_workers:
        for f in flist:
            if f.suffix == '.feather':
                feather_dict[f.stem.lower()] = load_trade_feather(f, index_col=index_col)[1]
    else:
        from multiprocessing import Pool
        p = Pool(n_workers)
        load_feather_par = partial(load_trade_feather, index_col=index_col)
        for res in p.imap_unordered(load_feather_par, flist):
            feather_dict[res[0]] = res[1]
    if pairs is not None:
        missing_pairs = set(pairs) - set(list(feather_dict.keys()))
    if len(missing_pairs) > 0:
        raise ValueError(f"Wasn't able to find the following requested pairs: {', '.join(list(missing_pairs))}")
    return feather_dict


if __name__ == '__main__':
    trade_dict = load_trades_feather_dir('~/fin_data/binance_trades/')
    print(trade_dict)
