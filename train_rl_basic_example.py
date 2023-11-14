from tsrl.environments import generate_candle_features
from tsrl.experiments.market.experiment_no_distillation import MarketExperiment1
from pathlib import Path
from plotly import graph_objects as go
import numpy as np
from tsrl.utils import memoize

# Add this 2 lines of commands
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1) Let's load data and generate features
# coin_list = ['EURCHF']
# coin_list = ['AUDCAD', 'EURCHF']
# coin_list = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD',
#              'EURCHF', 'EURDKK', 'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK', 'EURUSD', 'GBPAUD', 'GBPCAD',
#              'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'SGDJPY', 'USDCAD',
#              'USDCHF', 'USDDKK', 'USDHKD', 'USDJPY', 'USDNOK', 'USDSEK', 'USDSGD']
# PATH_TO_FEATHER = '../data/Forex/'

# let's use crypto for now
# coin_list = ['BTCUSDT', 'ETCUSDT', 'ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'LTCUSDT']
coin_list = ['BTCUSDT']
# PATH_TO_FEATHER = '../data/Crypto/minute_binance'
# PATH_TO_FEATHER = 'sample_data/'
PATH_TO_FEATHER = '/Users/georg/Source/Github/Thesis/financial-trading-in-RL/data/minute_binance/'
index_col = 'date'
# RESAMPLE = '30T'
resample = '1H'
train_end = "2021-03-14"


"""
Data is a dictionary for each asset  with keys: 
['asset_index', 'candle_dict', 'feature_dict', 'candle_df_dict']
"""
data = generate_candle_features(train_end=train_end,
                                pairs=coin_list,
                                feather_folder=PATH_TO_FEATHER,
                                timescale=resample,
                                feature_config=(
                                      # basic features
                                      dict(name='int_bar_changes', func_name='inter_bar_changes',
                                           columns=['close', 'high', 'low'],
                                           use_pct=True),
                                      # dict(name='int_bar_changes_10', func_name='inter_bar_changes',
                                      #      columns=['close', 'high', 'low'], use_pct=True,
                                      #      smoothing_window=10),
                                      # dict(func_name='hl_to_pclose'),
                                      # dict(func_name='next_return'),  # this is a cheat feature, to test we can learn
                                ))
print(data)
# 2) Let's set parameters for our environment
env_params = dict(
        max_episode_steps=40,
        commission_punishment=2e-5,
    )


# 3) Let's set parameters for our model (LSTM)
net_size = 32
model_params = dict(
    combine_policy_value=False,
    nb_actions=3,
    lstm_size=net_size,
    actor_size=[net_size],
    critic_size=[net_size],
    dropout=0.2
)

fig = go.FigureWidget()

for i in range(3):
    # 4) Now we can initialize our experiment
    exp_path = Path('experimentt').expanduser()
    # You can use sentiment data too -> use_sentiment = True
    exp = MarketExperiment1(exp_path=exp_path, use_sentiment=False)
    # 5) It's time to train our agent using specific parameters
    exp.train(data, model_params=model_params, env_params=env_params, train_range=("2000","2021-03-14"),
                  ppo_clip=0.2, n_envs=128, n_reuse_value=1,use_amp=False,rew_limit=6.,truncate_bptt=(5,20),
                  tau=0.95, env_step_init=1.0,
                  n_epochs=1000, validation_interval=502, show_progress=True, weight_decay=0.,
                  entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                  lr=5e-4, lookahead=False,
                  #advantage_type='exponential',
                  advantage_type='direct_reward',
                  gamma=0.99,
                  n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None)


    # 6) Let's check the agent's performance
    eval_dict = exp.eval(data)

    detailed_pnls = exp.detailed_backtest(data, eval_dict, train_end=train_end)
    for pair, pnls in detailed_pnls.items():
        train_pnl = pnls['train_pnl'].cumsum()
        test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
        fig.add_scatter(x=train_pnl.index,y=train_pnl, legendgroup=pair, name=pair)
        fig.add_scatter(x=test_pnl.index,y=test_pnl, legendgroup=pair, name=pair)

fig.show()

fig.write_html(f'{exp_path}/figure.html')

