from plotly import graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Add this 2 lines of commands
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

PATH_EXPERIMENTS = 'saved_models_experiments/'
PATH_FIGURES = 'experiments_figures/'


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def visualize(detailed_pnls, title, label, scaler):
    for pair, pnls in detailed_pnls.items():
        train_pnl = pnls['train_pnl'].cumsum()
        test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
        plt.title(title)
        plt.plot( train_pnl, label=label, color=adjust_lightness('r', scaler))
        plt.plot( test_pnl, color=adjust_lightness('r', scaler))
        # plt.plot(train_pnl + i, label=str(b), color=c)
        # plt.plot(test_pnl + i, label=str(b), color=c)
    plt.legend()


def visulazise_chrome(fig, detailed_pnls, legend=None):

    i = 0
    for pair, pnls in detailed_pnls.items():
        train_pnl = pnls['train_pnl'].cumsum()
        test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
        fig.add_scatter(x=train_pnl.index, y=train_pnl, legendgroup=pair, name=pair)
        fig.add_scatter(x=test_pnl.index, y=test_pnl, legendgroup=pair, name=pair)
    # for pair, pnls in detailed_pnls.items():
    #     train_pnl = pnls['train_pnl'].cumsum()
    #     test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
    #     fig.add_scatter(x=train_pnl.index, y=train_pnl, legendgroup=pair, name=legend)
    #     fig.add_scatter(x=test_pnl.index, y=test_pnl, legendgroup=pair, name=legend)
    #     i = i + 1

def visualize_chrome_avg_pnl(fig, data, name, pot, saved=False):
    # colors = ['#FF0000', '#C6DBEF', '#9ECAE1','#6BAED6', '#4292C6', '#08519C', '#041E5F']
    # colors =['#FF0000', '#CCCCCC', '#1E90FF', '#000000','#0000FF','#00008B']
    # colors = ['#FF0000', '#1E90FF',  '#000000']
    colors = ['#FF0000',  '#000000']
    if saved:
        fig.add_scatter(x=data.index, y=data['Value'], name=name)
        # fig.add_scatter(x=data.index, y=data['Value'], name=name,  marker= {'color': colors[pot]})

    else:
        fig.add_scatter(x=data.index, y=data, name=name, marker={'color': colors[pot]})



def visualize_avg(train_pnl, test_pnl, title, label, scaler):
    plt.title(title)
    plt.plot(train_pnl, label=label, color=adjust_lightness('r', scaler))
    plt.plot(test_pnl, color=adjust_lightness('r', scaler))
    plt.legend()


def visualize_test(test_pnl, title, label, scaler):
    colors =['#FF0000', '#CCCCCC', '#1E90FF', '#0000FF', '#000000','#00008B']

    plt.title(title)
    # plt.plot(test_pnl, label=label, color=adjust_lightness('r', scaler))
    plt.plot(test_pnl, label=label, color=colors[scaler])
    plt.ylabel('Pnl')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend()


def visualize_avg_runs(pnl, title, label, scaler):
    plt.title(title)
    plt.plot(pnl, label=label, color=adjust_lightness('r', scaler))
    plt.legend()


def visualize_distribution(data, title):
    # data = data.to_numpy()
    sns.set_theme()
    dict = {'Buy': data[data == 1.0].size, 'Sell': data[data == -1.0].size, 'Exit': data[data == 0.0].size}
    fig = plt.figure(figsize=(12,8))

    plt.title(title)
    plt.bar('Sell', height=dict['Sell'], color='r', label='Sell')
    plt.bar('Exit', height=dict['Exit'], color='b', label='Exit')
    plt.bar('Buy', height=dict['Buy'], color='g', label='Buy')
    plt.xlabel('Position of the agent')
    plt.ylabel('No. of position')
    plt.legend()
    plt.savefig(PATH_FIGURES+title)
    plt.show()


def visualize_signals(data, action_data, title='Agent actions'):
    sns.set_theme()
    fig = plt.figure(figsize=(12, 8))
    plt.title(title)
    # plt.plot(data, color='r', lw=2.)
    # plt.plot(data['Adj Close'], '^', markersize=10, color='m', label='buying signal', markevery=(data['Label']==2))
    # plt.plot(data['Adj Close'], 'v', markersize=10, color='k', label='selling\shorting signal', markevery=(data['Label']==1))

    plt.plot(data['2021-03-15 00:00:00':]['close'], marker='^', markersize=10, color='m', label='buying signal', markevery=(action_data[action_data == 1.0]))
    plt.plot(data['2021-03-15 00:00:00':]['close'], marker='v', markersize=10, color='k', label='selling signal', markevery=(action_data[action_data == -1.0]))

    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    # plt.savefig('Buy selling signals')
    plt.show()


def visualize_asset(data, asset):

    fig = go.FigureWidget()
    fig.add_scatter(x= data[:'2021-03-14'].index, y= data[:'2021-03-14']['close'],legendgroup=asset, name=asset)
    fig.add_scatter(x= data['2021-03-14':].index, y= data['2021-03-14':]['close'],legendgroup=asset, name=asset)

    fig.write_html(f'figure.html')
    fig.show()


def visualize_std(fig, data, name, pot, saved=False):
    # colors = ['#FF0000', '#C6DBEF', '#9ECAE1','#6BAED6', '#4292C6', '#08519C', '#041E5F']
    colors =['#FF0000', '#CCCCCC', '#1E90FF', '#0000FF', '#00008B', '#000000']
    colors =['#FF0000', '#CCCCCC', '#1E90FF', '#0000FF', '#000000']


    # fill_colors = ['RGB(255,0,0, 0.3)', 'RGB(204,204,204, 0.3)', 'RGB(30,144,255, 0.3)', 'RGB(0,0,255, 0.3)', 'RGB(0,0,139,0.3)']
    fill_colors = ['RGB(255,106,106, 0.3)', 'RGB(224,224,224, 0.3)', 'RGB(0,191,255, 0.3)', 'RGB(0,0,238, 0.3)', 'RGB(138,43,226,0.3)']

    # colors= ['#8B0000', '#008B00','#041E5F']
    # fill_colors =['rgba(139,0,0,0.3)', 'rgba(0,205,0,0.3)', 'rgba(0,0,139,0.3)']
    # fig.add_scatter(name=name,
    #                 x=data.index,
    #                 y=data['Value'],
    #                 marker= {'color': colors[pot]})

    fig.add_scatter(name=name,
                    x=data.index,
                    y=data['High'],
                    line=dict(width=0),
                    mode='lines',
                    showlegend=False
                    )

    fig.add_scatter(name=name,
                    x=data.index,
                    y=data['Low'],
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=fill_colors[pot],
                    fill='tonexty',
                    showlegend=False

                    )
    fig.add_scatter(name=name,
                    x=data.index,
                    y=data['Value'],
                    mode='lines',
                    line={'color': colors[pot]})
