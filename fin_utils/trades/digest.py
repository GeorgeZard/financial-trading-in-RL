import pandas as pd
from pathlib import Path
from tqdm import tqdm

from fin_utils.trades.loading import group_ticks
from fin_utils.preprocessing import transform_ticks_to_candles
from fin_utils.candles import resample_candles
import gc


def digest_binance(store_path, out_folder, digest_candles=True, resample='1h'):
    """
    Digests Binance trade data HDFStore to feathers.
    :param store_path:
    :param out_folder:
    :return:
    """
    symbols = requests.get("https://api.binance.com/api/v3/exchangeInfo").json()['symbols']
    symbols = {s['symbol']: s for s in symbols}
    out_folder = Path(out_folder).expanduser()
    trades_path = out_folder / 'trades'
    trades_path.mkdir(exist_ok=True, parents=True)
    candles_path = out_folder / 'candles'
    candles_path.mkdir(exist_ok=True, parents=True)
    row_num_dict = dict()
    with pd.HDFStore(store_path, 'r') as store:
        keys = list(store.keys())
        for k in keys:
            row_num_dict[k] = store.get_storer(k).nrows

    keys = sorted(keys, key=lambda k: row_num_dict[k], reverse=True)

    chunksize = 10_000_000
    for k in tqdm(keys):
        k = k.replace('/', '')
        if k not in symbols:
            continue
        sym = symbols[k]
        tick_size = float(next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', sym['filters']))['tickSize'])
        try:
            with pd.HDFStore(store_path, 'r') as store:
                dft = store.select(k, columns=['q', 'p', 'T', 'm'])
                # dft.loc[:, 'q'] = pd.to_numeric(dft['q'])
                # dft.loc[:, 'p'] = pd.to_numeric(dft['p'])
                dft.loc[:, 'T'] = pd.to_datetime(dft['T'], unit='ms')
        except Exception as ex:
            print(ex)
            continue
        gc.collect()
        dft.rename(columns=dict(
            T='date', q='quantity', m='side', p='price'
        ), inplace=True)
        print(f"Buidling candles for {k}")
        candle_list = []
        for df in tqdm(batch(dft, chunksize)):
            candle_list.append(transform_ticks_to_candles(df, window=resample))

        candles = pd.concat(candle_list, axis=0, copy=False)
        candles = resample_candles(candles, window=resample)
        candles.reset_index().to_feather(str(candles_path / f"{k.lower()}.feather"),
                                         # compression='zstd', compression_level=5
                                         )
        del candles, candle_list
        gc.collect()
        print(f"Candles built for {k}")
        if dft.shape[0] > chunksize:
            df_list = []
            for df in tqdm(batch(dft, chunksize)):
                df = group_ticks(df, time_group_span=resample, price_group_span=100 * tick_size,
                                 time_col='date', price_col='price', side_col='side', size_col='quantity')
                df_list.append(df)
            df = pd.concat(df_list, axis=0, copy=False)
            del df_list
        else:
            df = dft
            df = group_ticks(df, time_group_span=resample, price_group_span=100 * tick_size,
                             time_col='date', price_col='price', side_col='side', size_col='quantity')
        df.reset_index(drop=True, inplace=True)

        # for Binance 'm' is:
        # "m": true -> Was the buyer the maker?
        # but we want "side": True if the buyer was taker.
        df.loc[:, 'side'] = ~df.loc[:, 'side']
        df.to_feather(str(trades_path / f"{k}.feather"),
                      compression='zstd', compression_level=5)
        del df
        gc.collect()

    # for pair_path in tqdm(list(trades_path.iterdir())):
    #     pair = pair_path.stem
    #     df = pd.read_feather(pair_path)
    #     candles = transform_ticks_to_candles(df, window=resample)
    #     candles.reset_index().to_feather(str(candles_path / f"{pair}.feather"),
    #                                      compression='zstd', compression_level=5)
    #     del candles
    #     del df
    #     gc.collect()


def digest_coinbase(store_path, out_folder):
    raise NotImplemented()


def batch(iterable, batch_number=10):
    """
    split an iterable into mini batch with batch length of batch_number
    supports batch of a pandas dataframe
    usage:
        for i in batch([1,2,3,4,5], batch_number=2):
            print(i)

        for idx, mini_data in enumerate(batch(df, batch_number=10)):
            print(idx)
            print(mini_data)
    """
    l = len(iterable)

    for idx in range(0, l, batch_number):
        if isinstance(iterable, pd.DataFrame):
            # dataframe can't split index label, should iter according index
            yield iterable.iloc[idx:min(idx + batch_number, l)]
        else:
            yield iterable[idx:min(idx + batch_number, l)]


import requests
import time, pickle


def digest_bitmex(store_path, out_folder, resample='1H'):
    instr_pkl_path = Path('./bitmex_instr_details.pkl').expanduser()
    if instr_pkl_path.exists():
        with open(instr_pkl_path, 'rb') as f:
            instr_dict = pickle.load(f)

    else:
        instr_details_new = requests.get('https://www.bitmex.com/api/v1/instrument?count=500&reverse=false').json()
        time.sleep(0.3)
        instr_details = instr_details_new
        start = 0
        while len(instr_details_new) > 0:
            instr_details_new = requests.get(
                f'https://www.bitmex.com/api/v1/instrument?count=500&reverse=false&start={len(instr_details)}').json()
            instr_details.extend(instr_details_new)
            if len(instr_details_new) < 500:
                break
            time.sleep(0.3)
        instr_dict = {inst['symbol']: inst for inst in instr_details}
        with open(instr_pkl_path, 'wb') as f:
            pickle.dump(instr_dict, f)

    out_folder = Path(out_folder).expanduser()
    trades_path = out_folder / 'trades'
    trades_path.mkdir(exist_ok=True, parents=True)
    candles_path = out_folder / 'candles'
    candles_path.mkdir(exist_ok=True, parents=True)
    with pd.HDFStore(store_path, 'r') as store:
        keys = list(store.keys())

    chunksize = 10_000_000
    for k in tqdm(keys):
        k = k.replace('/', '')
        gc.collect()
        with pd.HDFStore(store_path, 'r') as store:
            # if k not in instr_dict:
            #     continue
            nrows = store.get_storer(k).nrows
            start = 0
            if start < 0 and nrows < abs(start):
                print(f"{k} has {nrows} records. {start} requests more rows that availabe. Setting to 0.")
                start = 0
            n_req_rows = abs(start) if start < 0 else nrows - start
            tick_size = instr_dict[k]['tickSize']

            if n_req_rows < chunksize:
                df: pd.DataFrame = store.select(k, start=start)
                df = group_ticks(df, time_group_span=resample, time_col='time', size_col='size', side_col='side',
                                 price_col='price', price_group_span=tick_size * 10)
            else:
                df_list = []
                df: pd.DataFrame = store.select(k, start=start)
                for df_s in tqdm(batch(df, chunksize), total=nrows // chunksize):
                    df_s = group_ticks(df_s, time_group_span=resample, time_col='time', size_col='size',
                                       side_col='side',
                                       price_col='price', price_group_span=tick_size * 10)
                    df_list.append(df_s)
                # for df in tqdm(
                #         store.select(k, start=start, chunksize=chunksize, iterator=True), total=nrows // chunksize):
                #     # gc.collect()
                #     df = group_ticks(df, time_group_span=resample, time_col='time', size_col='size', side_col='side',
                #                      price_col='price', price_group_span=tick_size * 10)
                #     df_list.append(df)
                df = pd.concat(df_list, axis=0, copy=False)

        df.reset_index(drop=True, inplace=True)
        df.to_feather(str(trades_path / f"{k}.feather"),
                      compression='zstd', compression_level=3)
        del df

    for pair_path in tqdm(list(trades_path.iterdir())):
        pair = pair_path.stem
        df = pd.read_feather(pair_path)
        candles = transform_ticks_to_candles(df, window=resample, quantity_col='size', timestamp_col='time')
        candles.reset_index().to_feather(str(candles_path / f"{pair}.feather"))
        del candles
        del df
        gc.collect()


def digest():
    digest_binance(Path('~/fin_data/binance_ticks_large_chunks.h5').expanduser(),
                   Path('~/fin_data/binance/').expanduser(), resample='1H')


# digest_bitmex(Path('~/fin_data/bitmex_ticks.h5').expanduser(),
#               Path('~/fin_data/bitmex/').expanduser()
#               )


if __name__ == '__main__':
    digest()
