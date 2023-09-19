"""
This module contains simple functions used to calculate the cumulative percentage profits
of a set of trading decisions in a given OHLC timeseries. Some major assumptions exist in
these methods which we explain for each function.

Here is an example calculating the percentage pnl. The timeseries is simplified to a
single value for each timestep (i.e. close == open). The numbers chosen are purposefully
different from real prices to help you build intuition. Commission in this example is 0.

'''
time      t1   t2   t3   t4   t5   t6   t7   t8
-------------------------------------------------
price     10   10   20   30   40   30   20   40
-------------------------------------------------
position   0    1    1    0   -1   -1   -1    0
-------------------------------------------------
pnl        0    0    1.  1.   0  .25  .25  -0.5
'''

At time t2 the position is set to 1 (long) continues until t4 where the position changes
to 0. From t2 -> t3 the price changes from 10 to 20 which is a 100% return. Then again from
t3 -> t4 the price changes from 20 to 30, but since the position was opened at price 10 the return
is still 100% ((30 - 20) / 10) instead of 50% (which would occur if the long position was first opened
at t3 instead of t2) . From t4 -> t5 there is no return since the position at t4 is 0. Then
from t5 -> t6 the price drops from 40 to 30 and since the position is -1 (short) we profit yet again
with 25% return. Similarly from t6 -> t7 the price drops and we profit 25% return (instead of 33%
if the short position was first opened at t6). Finally from t7 -> t8 the price rises back up to 40
and all the profit accumulated from the short position is wiped out yielding -50% return. Had we
chosen a position 0 at timestep t7 the profit of the short position would stay at 50% cumulatively.

An important concept is also the realized vs unrealized PnL. For the example above here are the
two values through time.
'''
time               t1   t2   t3   t4   t5   t6    t7    t8
-------------------------------------------------------------
realized PnL       0    0    0    2.   2.   2.    2.    2.
-------------------------------------------------------------
unrealized PnL     0    0    1.   2.   2.   2.25  2.50  2.
'''

The unrealized PnL is the cumulative sum of the pnl throughout the open positions while the realized pnl
changes only when a position is closed and the profits or losses are solidified.


The most important assumptions for Profit and Loss (PnL) calculated here are:
- *No reinvesting / Constant trading lot* : To avoid defining the size of each trade we
  consider an arbitrary constant sized lot for each one. Whenever a position changes
  the lot size does not change whether profit or loss has accumulated up to this point.
  This provides an unbiased view of the strategies performance across time. When the
  accumulated profit or loss is included in the lot size (i.e. when you add the amount
  of profit or subtract the loss from your trading size) the behaviour across time can
  differ significantly across strategies. By keeping the lot constant strategies can
  be directly compared throughout the timeseries without needing to convert to percentages.
  If the actual pnl from reinvesting the profits or subtracting the loss is needed, you can
  always do `(pnl+1.).cumprod` instead of what we do which is `pnl.cumsum()`.
- *No separate slippage consideration* : To avoid unnecessary complication we keep both the
  slippage and commission as a single commission parameter. You can increase it if you
  wish to calculate the pnl with higher slippage.
- *Execution* : It is implied that the execution of the order happens on the open price of
  the next OHLC candle.

"""

import pandas as pd
import numpy as np
import warnings


def pnl_from_positions_detailed(candles: pd.DataFrame, positions: pd.Series, commission=0.) -> dict:
    # assert candles.shape[0] == positions.shape[0]
    assert type(candles.index) is pd.DatetimeIndex
    assert type(positions.index) is pd.DatetimeIndex
    cshp, pshp = candles.shape[0], positions.shape[0]
    candles = candles.iloc[:candles.index.searchsorted(positions.index[-1]) + 1]
    candles, positions = candles.align(positions, axis=0, join='left', method='ffill')
    assert pshp <= positions.shape[0], "Some positions are getting dropped because they " \
                                       "are not aligned with the provided candles."
    pos_changes = abs(positions.diff().fillna(positions))
    pos_prices = candles.open.copy()
    pos_prices[~(pos_changes > 0)] = pd.NA
    pos_prices = pos_prices.ffill().fillna(candles.open)
    comm_charges = pos_changes * commission
    candle_returns = (candles.close - candles.open) / pos_prices
    step_returns = (candles.open.shift(-1).fillna(candles.open) - candles.close) / pos_prices
    total_returns = (candle_returns + step_returns) * positions
    pnl = total_returns - comm_charges
    n_trades = pos_changes.sum()
    time_span = (pnl.index[-1] - pnl.index[0]) / pd.to_timedelta('360D')
    return dict(pnl=pnl, n_trades=n_trades, annualized_pnl=pnl.sum() / time_span,
                annualized_n_trades=n_trades / time_span, positions=positions)


def pnl_from_positions(candles: pd.DataFrame, positions: pd.Series, commission=0.) -> pd.Series:
    """
    Calculate the percentage pnl gained given a set of candles and the positions an agents holds.
    We assume that position changes happen using market orders which execute at next open.

    ..    t1    t2    t3   (timesteps)
    --------------------------
    ..     o     o     o   (candle values open/high/low/close)
    ..     h     h     h
    ..     l     l     l
    ..     c     c     c

    if the agent/trader decides to buy/sell with information up to (and including) t1,
    then the order is executed at the open price of t2 (if market, or in the range low-high
    of t2 if limit). This is because at the time the trader knows the prices in t1 it is
    already too late to execute the trade. This means that if a position is decided using
    information at index t1, the position array index must be t2. Otherwise you might
    be calculating the profit of positions based on future information.

    ** IMPORTANT: Make sure the provided positions are at t2 if your prediction is using
                  information up to and including t1. This can be done with positions.shift()
                  to push the positions one index forward in time.

    :param candles: Dataframe of price candles. Must have a datetime indexes to be able
                    to align with the positions.
    :param positions: Series of positions. Position take values from {0,1,-1}, where
                      out of market = 0, buy = 1, sell = -1. Must have a datetime indexes.
    :param commission: The commission percentage for when the price changes.
    :return: pnl
    """
    bt_res = pnl_from_positions_detailed(candles, positions, commission)
    return bt_res['pnl']


def pnl_from_price_position(candles: pd.DataFrame, trade_price: pd.Series,
                            positions: pd.Series, commission=0.):
    # assert candles.shape[0] == positions.shape[0] == trade_price.shape[0]
    assert type(candles.index) is pd.DatetimeIndex
    assert type(positions.index) is pd.DatetimeIndex
    assert type(trade_price.index) is pd.DatetimeIndex
    cshp, pshp = candles.shape[0], positions.shape[0]
    candles, positions = candles.align(positions, axis=0, join='left', method='ffill')
    assert pshp <= positions.shape[0], "Some positions are getting dropped because they " \
                                       "are not aligned with the provided candles."
    candles, trade_price = candles.align(trade_price, axis=0, join='left', method='ffill')
    assert positions.shape[0] == trade_price.shape[0], "positions must be the same size as trade_price"
    pos_changes = abs(positions.diff().fillna(positions))
    pos_prices = trade_price.copy()
    pos_prices[pos_changes == 0] = pd.NA
    pos_prices = pos_prices.ffill().fillna(trade_price)
    comm_charges = pos_changes * commission
    # candle_returns = (candles.close - trade_price) / pos_prices
    # step_returns = (trade_price - candles.open.shift().fillna(candles.open)) / pos_prices
    # total_returns = ((candle_returns + step_returns) * positions).fillna(0.)
    total_returns = positions * (trade_price.shift(-1).ffill() - trade_price) / pos_prices
    pnl = total_returns - comm_charges
    return pnl.fillna(0.)


def pnl_from_limit_orders(candles: pd.DataFrame,
                          orders: pd.Series,
                          limits: pd.DataFrame,
                          commission=0.):
    assert candles.shape[0] == orders.shape[0]
    assert type(candles.index) is pd.DatetimeIndex
    assert type(orders.index) is pd.DatetimeIndex
    assert type(limits)
    raise NotImplemented()
