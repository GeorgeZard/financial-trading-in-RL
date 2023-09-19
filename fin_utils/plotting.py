from plotly import graph_objs as go
from plotly.offline import plot
from plotly import tools
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd

from typing import Union, Dict
import pandas as pd
import numpy as np
import colorlover as cl


def get_colorscale(num_bins: int,cl_scale='BuPu'):
    bupu = cl.scales['9']['seq'][cl_scale]
    return cl.interp(bupu, num_bins)


def get_reduced_hsl_transparency(hsl_string, transparency=0.3):
    if len(hsl_string.split(',')) > 3:
        raise NotImplementedError()
    else:
        hsl_string.replace(')', f', {transparency})')


tight_layout_kwargs = dict(layout=go.Layout(margin=go.layout.Margin(l=45, r=30, b=40, t=30, pad=4)))


def create_candle_figure(candles: pd.DataFrame, slider=False, figure=None):
    if figure is None:
        figure = go.Figure(
            layout=go.Layout(
                margin=go.layout.Margin(
                    l=45,
                    r=30,
                    b=40,
                    t=30,
                    pad=4
                )))
    figure.add_candlestick(
        x=candles.index,
        close=candles.close,
        open=candles.open,
        high=candles.high,
        low=candles.low,
        whiskerwidth=0,
        name='price candles'
    )
    figure.layout.xaxis['rangeslider']['visible'] = slider
    return figure


def position_figure(positions: pd.Series, candles):
    fig = create_candle_figure(candles)
    indexes = positions.index
    stay = indexes[positions == 0]
    up = indexes[positions == 1]
    down = indexes[(positions == 2) | (positions == -1)]

    fig.add_scatter(y=candles.open.loc[stay], x=stay, name='exit/stay', mode='markers',
                    marker=dict(color="black", symbol="x", size=15))
    fig.add_scatter(y=candles.open.loc[up], x=up, name='buy', mode='markers',
                    marker=dict(color="green", symbol="triangle-up-open", size=15))
    fig.add_scatter(y=candles.open.loc[down], x=down, name='sell', mode='markers',
                    marker=dict(color="red", symbol="triangle-down-open", size=15))
    return fig


def plot_market_limit_prices(
        actions, limit_prices, candles, title=None, filename=None,
        first=None, pnl=None, positions=None,
        auto_open=False
):
    candles = candles.loc[actions.index]
    # pnl = pnl.iloc[pnl.index.searchsorted(limit_prices.index[:first])]
    # positions = positions.iloc[positions.index.searchsorted(limit_prices.index[:first])]
    # limit_prices = limit_prices.iloc[:first]
    if actions.ndim == 2:
        actions = actions.iloc[:, 0]
    close_idxs = actions.index[actions == 0]
    market_buy_idxs = actions[actions == 1].index
    market_sell_idxs = actions[actions == 2].index
    limit_buy_idxs = actions[(actions == 3) | (actions == 5)].index
    limit_sell_idxs = actions[(actions == 4) | (actions == 5)].index
    # # print([np.count_nonzero(x) \
    # #        for x in [close_idxs, market_buy_idxs, market_sell_idxs, limit_buy_idxs, limit_sell_idxs]])
    #
    buy_limits = limit_prices.loc[limit_buy_idxs, 'buy_limit'].copy()
    sell_limits = limit_prices.loc[limit_sell_idxs, 'sell_limit'].copy()
    fig = make_subplots(rows=3, cols=1, row_heights=[0.8, 0.1, 0.1],
                        shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05)
    fig.update_layout(title=title)
    upper_kw = dict(row=1, col=1)
    # fig = go.Figure()
    # upper_kw = dict()
    fig.add_candlestick(x=candles.index,
                        close=candles.close, open=candles.open,
                        high=candles.high, low=candles.low,
                        **upper_kw)
    # fig.add_scattergl(x=candles.index, y=candles.high, name='High', **upper_kw)
    # fig.add_scattergl(x=candles.index, y=candles.low, name='Low', **upper_kw)
    # fig.add_scattergl(x=candles.index, y=candles.close, name='Close', **upper_kw)
    fig.add_scattergl(
        y=candles.close.loc[close_idxs].values,
        x=close_idxs,
        name="close",
        mode="markers",
        marker=dict(color="black", symbol="x", size=15), **upper_kw
    )
    fig.add_scattergl(
        y=candles.open[market_buy_idxs].values,
        x=market_buy_idxs,
        name="market buy",
        mode="markers",
        marker=dict(color="green", symbol="triangle-up-open", size=15), **upper_kw
    )
    fig.add_scattergl(
        y=candles.open[market_sell_idxs],
        x=market_sell_idxs,
        name="market sell",
        mode="markers",
        marker=dict(color="red", symbol="triangle-down-open", size=15), **upper_kw
    )
    fig.add_scattergl(
        y=buy_limits,
        x=buy_limits.index,
        name="limit buy",
        mode="markers",
        marker=dict(color="green", symbol="triangle-up", size=15), **upper_kw
    )
    fig.add_scattergl(
        y=sell_limits,
        x=sell_limits.index,
        name="limit sell",
        mode="markers",
        marker=dict(color="red", symbol="triangle-down", size=15), **upper_kw
    )
    positive_pnl = pnl[pnl > 0]
    negative_pnl = pnl[pnl <= 0]
    fig.add_trace(
        go.Bar(x=positive_pnl.index, y=positive_pnl, name='Gains', marker_color='green'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=negative_pnl.index, y=negative_pnl, name='Losses', marker_color='red'),
        row=2, col=1
    )
    if positions is not None:
        ups = positions[positions == 1]
        downs = positions[positions == -1]
        fig.add_trace(go.Bar(x=ups.index, y=ups,
                             name='Longs', marker_color='green'), row=3, col=1)
        fig.add_trace(go.Bar(x=downs.index, y=downs,
                             name='Shorts', marker_color='red'), row=3, col=1)

    plot(fig, filename=filename, auto_open=auto_open)


def write_figure_to_file(figure: go.Figure, filename: Union[str, Path]):
    filepath = Path(filename)
    if filepath.suffix == '.html':
        figure.write_html(str(filename))
    else:
        figure.write_image(str(filename))


def plot_candles(candles: pd.DataFrame, filename):
    figure = create_candle_figure(candles)
    write_figure_to_file(figure, filename)


def get_df_traces_with_error(df_list, name, color=None):
    df = pd.concat(df_list, axis=1).ffill()
    # df = df.iloc[:,np.argsort(df.sum().values)[-15:]]
    error = df.std(axis=1).rolling(21, min_periods=1, center=True).mean()
    mean = df.mean(axis=1)
    fillcolor = get_reduced_hsl_transparency(color, 0.05)
    # transparent = get_reduced_hsl_transparency(color, 0.)
    plt_kwgs = dict(fillcolor=fillcolor, legendgroup=name, name=name)
    main_trace = go.Scatter(x=mean.index, y=mean,
                            fill='tonexty',
                            line=dict(color=color),
                            **plt_kwgs)
    upper_trace = go.Scatter(x=mean.index, y=mean + error,
                             fill='tonexty',
                             showlegend=False,
                             line=dict(color=color, width=0),
                             **plt_kwgs)
    lower_trace = go.Scatter(x=mean.index, y=mean - error,
                             showlegend=False,
                             opacity=0.5,
                             line=dict(color=color, width=0),
                             **plt_kwgs)
    return [lower_trace, main_trace, upper_trace]
