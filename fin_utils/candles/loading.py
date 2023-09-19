from functools import partial
import pandas as pd
from pathlib import Path
from fin_utils.candles import resample_candles
import warnings

def load_feather(path: Path, resample=None, index_col='date'):
    df = pd.read_feather(str(path))
    cols = df.columns.tolist()
    for c in ['open', 'high', 'low', 'close', index_col]:
        assert c in cols, f"{c} is missing from columns from {str(path)}. All cols in this df is {cols}"
    if index_col:
        df.set_index(index_col, inplace=True)
    if resample:
        df = resample_candles(df, window=resample)
    return path.stem.lower(), df


def load_feather_dir(path: Path, resample=None, pairs=None, index_col='date', n_workers=None):
    path = Path(path).expanduser()
    feather_dict = dict()
    dirlist = list(path.iterdir())
    flist = [f for f in dirlist if f.suffix == '.feather']
    if len(flist) < len(dirlist):
        ignored_files = set(dirlist) - set(flist)
        ignored_files = [f.name for f in ignored_files]
        print(f"Ignoring non-feather files: {','.join(ignored_files)}")
    if pairs is not None:
        pairs = [p for p in pairs]
        flist = filter(lambda f: f.stem in pairs, flist)
    if not n_workers:
        for f in flist:
            if f.suffix == '.feather':
                feather_dict[f.stem] = load_feather(f, resample=resample, index_col=index_col)[1]
    else:
        from multiprocessing import Pool
        p = Pool(n_workers)
        load_feather_par = partial(load_feather, resample=resample, index_col=index_col)
        for res in p.imap_unordered(load_feather_par, flist):
            feather_dict[res[0]] = res[1]
    for k in list(feather_dict.keys()):
        zero_prices = feather_dict[k][['open', 'high', 'low', 'close']] == 0
        if zero_prices.any().any():
            df = feather_dict[k]
            feather_dict[k] = df[~zero_prices.any(axis=1)]
            warnings.warn(
                f"Found close price zero in candles of {k} pair. Dropping zeros... {feather_dict[k].shape[0]}")
    return feather_dict


if __name__ == '__main__':
    candle_dict = load_feather_dir('~/financial_data/histdata_feathers/')
    print(candle_dict)
