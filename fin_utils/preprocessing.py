import pandas as pd
from typing import Union


def transform_ticks_to_candles(ticks: pd.DataFrame, window: Union[str, pd.Timedelta] = '1H', drop_nans=False,
                               quantity_col='quantity', timestamp_col='date'):
    """
    Tranforms a dataframe of tick data into a candle dataframe

    :param ticks: a dataframe of ticks. It must contain the columns 'quantity', 'date' (or the ones defined as keyword
                  arguements quantity_col and timestamp_col). If the quantity_col column is available
                  the resulting candles will also contain a volume column for each window interval.
    :param window: the wi
    :param drop_nans:
    :return:
    """

    assert all([c in ticks.columns for c in [timestamp_col, 'price']])
    # ticks = ticks.copy()

    # ticks.time = pd.to_datetime(ticks.time)
    # ticks.price = ticks.price.astype(np.float32)
    # ticks[quantity_col] = ticks[quantity_col].astype(np.float32)
    ticks = ticks.set_index(timestamp_col)

    price_resampler = ticks.resample(window, label='right',
                                     closed='right')['price']

    agg_dict = dict(
        open=price_resampler.first(),
        high=price_resampler.max(),
        low=price_resampler.min(),
        close=price_resampler.last(),
    )
    if quantity_col in ticks.columns:
        volume_resampler = ticks.resample(window, label='right',
                                          closed='right')[quantity_col]
        agg_dict['volume'] = volume_resampler.sum()

    candles = pd.DataFrame(data=agg_dict)
    if drop_nans:
        candles = candles.dropna()
    else:
        nans = pd.isnull(candles.close)
        candles.close.ffill(inplace=True)
        candles.loc[nans, 'high'] = candles.close[nans]
        candles.loc[nans, 'low'] = candles.close[nans]
        candles.loc[nans, 'open'] = candles.close[nans]

    if 'volume' in candles.columns:
        candles['volume'].fillna(0, inplace=True)

    return candles
