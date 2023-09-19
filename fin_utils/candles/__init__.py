import pandas as pd


def resample_candles(candles: pd.DataFrame, window, label="right", closed="right", dropna=True) -> pd.DataFrame:
    if not isinstance(candles.index, (pd.DatetimeIndex)):
        raise ValueError("Candle dataframe index is not a Datetimeindex")
    aggregation_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in candles.columns:
        aggregation_dict["volume"] = "sum"
    candles = candles.resample(window, label=label, closed=closed).agg(aggregation_dict).dropna()
    if dropna == True:
        return candles.dropna()
    return candles


def cast_column_to_time(col: pd.Series):
    if isinstance(col.iloc[0], (pd.DatetimeIndex, pd.DatetimeScalar, pd.Timestamp)):
        return col
    else:
        return pd.to_datetime(col)


def get_time_index(candles: pd.DataFrame):
    if 'date' in candles.columns:
        return cast_column_to_time(candles['date'])
    elif 'time' in candles.columns:
        return cast_column_to_time(candles['time'])
    else:
        if isinstance(candles.index, (pd.DatetimeIndex)):
            return candles.index
        else:
            raise ValueError("Candle Dataframe does not seem to have any time index")
