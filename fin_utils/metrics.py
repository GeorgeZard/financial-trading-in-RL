import pandas as pd


def percentage_volatility(candles: pd.DataFrame, avg_window, future=False, center=False):
    assert not (future and center)
    sl = slice(None, None, -1) if future else slice(None, None, None)
    c_vols = (candles.open / candles.low + candles.high / candles.open) * 0.5 - 1.
    c_vols = c_vols.iloc[sl].rolling(window=avg_window, min_periods=1, center=center).mean().iloc[sl].ffill()
    return c_vols

