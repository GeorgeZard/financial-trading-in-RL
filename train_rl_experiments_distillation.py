import random
from tsrl.environments import generate_candle_features
from tsrl.experiments.market.experiment_no_distillation import MarketExperiment1
from tsrl.experiments.market.experiment_offline_dist import MarketExperiment2
from tsrl.experiments.market.experiment_online_dist import MarketExperiment3
from tsrl.experiments.market.experiment_online_pkt import MarketExperiment4
from tsrl.experiments.market.experiment_offline_pkt import MarketExperiment5
from tsrl.experiments.market.experiment_pkt_logit_dist import MarketExperiment6
from tsrl.experiments.market.experiment_new_online_dist import MarketExperiment7
from plotly.graph_objs import *
import csv
from pathlib import Path
from plotly import graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from fin_utils.candles.labels import construct_labels
import Visualizations

# Add this 2 lines of commands
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

PATH_EXPERIMENTS = 'saved_models_experiments/'

PATH_FIGURES = 'experiments_figures/'

PATH_DISTILLATION_NAMES = ['no_distillation/', 'offline_distillation2/', 'online_distillation/', 'online_pkt/', 'online_pkt_logits/', 'others/', 'offline_pkt/']

# PATH_EXPERIMENTS = 'saved_models_experiments_for_early_training/'
# PATH_FIGURES = 'experiments_figures_for_early_training/'

VISUALIZATIONS = Visualizations


def get_statistical_measures(data, eval_dict):
    # exp1 = eval_dict['BTCUSDT']['2021-05-05 00:00:00':'2021-08-10 00:00:00']
    exp1 = eval_dict['BTCUSDT']['2021-03-15 00:00:00':]
    # columns of exp is trade_price, transaction_price, position, reward
    # print(exp1['reward'])
    # exp1 = eval_dict['BTCUSDT']
    VISUALIZATIONS.visualize_distribution(exp1['position'], 'Bitcoin positions distribution for testing period ')
    # Visualizations.visualize_signals(data['candle_df_dict']['BTCUSDT'], exp1['position'], 'test')


# Adding title properly in FigureWidget
def add_title(fig, title, height):
    fig.update_layout(
        height=height,
        title=dict(
            text='<b>'+title+'</b>',
            x=0.5,
            y=0.95,
            font=dict(
                family="Arial",
                size=20,
                color='#000000'
            )
        ),
        xaxis_title="<b>epochs</b>",
        yaxis_title='<b>Profit and Loss Percentage</b>',
        font=dict(
            family="Courier New, Monospace",
            size=12,
            color='#000000'
        )
    )


def testing(data, start_training=False, seed=0, coin_list_size=26, train_end = "2021-03-14", env_params = dict, model_params = dict):
    exp_path1 = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[0] + 'experiment_no_distillation_many_assets_testing' + str(
        seed)).expanduser()
    exp1 = MarketExperiment1(exp_path=exp_path1, use_sentiment=False)

    n_epochs = 100
    val_interval = 10
    learning_rate = 5e-4
    n_teachers = 4
    net_size = 32
    start_training = True
    pnl_results = []
    if start_training:
        pnl_results = exp1.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                   ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                   tau=0.95, env_step_init=1.0,
                   n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                   entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                   lr=learning_rate, lookahead=False,
                   # advantage_type='exponential',
                   advantage_type='direct_reward',
                   gamma=0.99,
                   n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, seed=seed, write_to_tensorboard=False)

    print(pnl_results)


def test_distillations(data, start_training=False, seed=0, coin_list_size=26, train_end = "2021-03-14", env_params = dict, model_params = dict):

    no_dist_dict = []
    offline_dist_dict = []
    online_dist_dict = []
    online_pkt_dict = []
    online_log_pkt = []
    self_online = []
    self_pkt = []

    self_distillation = False
    scaler = [0.01, 0.6, 0.9, 1.1, 1.3, 1.5]
    # seed = 1
    # figure1 = plt.figure()
    # final_figure = plt.figure()

    runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seed = 0
    # runs = [7, 8, 9]
    mul_runs = False
    fig_avg_pnl = go.FigureWidget()
    add_title(fig_avg_pnl, 'Average test pnl for Different Types of Distillations', 850)
    # NO_STEP = 'no_step/'
    for i, item in enumerate(runs):
        print(f"------------------------------ Test {i+1} ------------------------------")
        seed = item
        # seed = seed
        fig1 = go.FigureWidget()
        # fig1.layout.title = 'No distillation'
        add_title(fig1, 'No distillation', 1200)

        fig2 = go.FigureWidget()
        # fig2.layout.title = 'Offline distillation'
        add_title(fig2, 'Offline distillation', 1200)

        fig3 = go.FigureWidget()
        # fig3.layout.title = 'Online distillation'
        add_title(fig3, 'Online distillation', 1200)

        fig4 = go.FigureWidget()
        # fig4.layout.title = 'Online PKT distillation'
        add_title(fig4, 'Online PKT distillation', 1200)

        fig5 = go.FigureWidget()
        # fig5.layout.title = 'Offline PKT distillation'
        add_title(fig5, 'Offline PKT distillation', 1200)

        fig6 = go.FigureWidget()
        # fig6.layout.title = 'Online PKT-Logits distillation'
        add_title(fig6, 'Online PKT-logits distillation', 1200)




        # # 4) Now we can initialize our experiment
        exp_path1 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[0]+'experiment_no_distillation_many_assets_testing' + str(seed)).expanduser()
        exp_path2 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[1]+'experiment_offline_distillation_many_assets_testing' + str(seed)).expanduser()
        exp_path3 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[2]+'experiment_online_distillation_many_assets_testing' + str(seed)).expanduser()
        exp_path4 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[3]+'experiment_pkt_many_assets_testing' + str(seed)).expanduser()
        exp_path5 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[6]+'experiment_offline_pkt_distillation_many_assets_testing'+ str(seed)).expanduser()
        exp_path6 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[4]+'experiment_pkt_logit_distillation_many_assets_testing'+ str(seed)).expanduser()

        # exp_path1 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[0] + 'experiment_no_distillation_many_assets' + str(seed)).expanduser()
        # exp_path2 = Path(PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[1] + 'experiment_offline_distillation_many_assets' + str(seed)).expanduser()
        # exp_path3 = Path(PATH_EXPERIMENTS + NO_STEP + PATH_DISTILLATION_NAMES[2] + 'experiment_online_distillation_many_assets' + str(seed)).expanduser()
        # exp_path4 = Path(PATH_EXPERIMENTS + NO_STEP + PATH_DISTILLATION_NAMES[3] + 'experiment_pkt_many_assets' + str(seed)).expanduser()
        # exp_path6 = Path(PATH_EXPERIMENTS + NO_STEP + PATH_DISTILLATION_NAMES[4] + 'experiment_pkt_logit_distillation_many_assets'+ str(seed)).expanduser()


        # You can use sentiment data too -> use_sentiment = True
        exp1 = MarketExperiment1(exp_path=exp_path1, use_sentiment=False)
        exp2 = MarketExperiment2(exp_path=exp_path2, use_sentiment=False)
        exp3 = MarketExperiment3(exp_path=exp_path3, use_sentiment=False)
        exp4 = MarketExperiment4(exp_path=exp_path4, use_sentiment=False)
        exp5 = MarketExperiment5(exp_path=exp_path5, use_sentiment=False)
        exp6 = MarketExperiment6(exp_path=exp_path6, use_sentiment=False)


        # list_expiriments = [exp1, exp2, exp3]
        # 5) It's time to train our agent using specific parameters

        n_epochs = 250
        beta = 10
        plt.figure(figsize=(12, 8))
        # sns.set_theme()
        scaler = [0.01, 0.6, 0.9, 1.1, 1.3, 1.5]
        val_interval = 5
        learning_rate = 5e-4
        n_teachers = 2
        net_size = 32
        start_training = True
        """
        No Distillation
        """
        print('--------------------- No Distillation ---------------------')
        if  start_training:
            exp1.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                      ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                      tau=0.95, env_step_init=1.0,
                      n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                      entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                      lr=learning_rate, lookahead=False,
                      # advantage_type='exponential',
                      advantage_type='direct_reward',
                      gamma=0.99,
                      n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, seed=seed)

        # 6) Let's check the agent's performance
        dir = PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[0]+'experiment_no_distillation_many_assets_testing'+str(seed)+'/exp_state_dict.pkl'
        print(dir)
        # data, exp, exp_path, fig, fig_avg_pnl, dir = '', train_end = '2022', coin_list_size = 17, save_models = [], mul_runs = False, scaler = []
        evaluation(data, exp1, exp_path1, fig1, fig_avg_pnl, dir, train_end, coin_list_size, no_dist_dict, mul_runs=mul_runs, scaler=scaler, scaler_pot=0)

        """
        Offline Distillation
        """
        print('--------------------- Offline Distillation ---------------------')
        # 3) Let's set parameters for our teacher model (LSTM)
        net_size = 128
        teacher_model_params = dict(
            combine_policy_value=False,
            nb_actions=3,
            lstm_size=net_size,
            actor_size=[net_size],
            critic_size=[net_size],
            dropout=0.2
        )

        if start_training:
            exp2.teacher_train(data, teacher_model_params=teacher_model_params, env_params=env_params,
                              train_range=("2000", "2021-03-14"),
                              ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                              tau=0.95, env_step_init=1.0,
                              n_epochs=n_epochs, validation_interval=502, show_progress=True, weight_decay=0.,
                              entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                              lr=learning_rate, lookahead=False,
                              # advantage_type='exponential',
                              advantage_type='direct_reward',
                              gamma=0.99,
                              n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=False, seed=seed, n_teachers=n_teachers)

            # 3) Let's set parameters for our student model (LSTM)
            net_size = 32
            model_params = dict(
                combine_policy_value=False,
                nb_actions=3,
                lstm_size=net_size,
                actor_size=[net_size],
                critic_size=[net_size],
                dropout=0.2
            )

            # print("Student Training")

            exp2.student_train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                              ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                              tau=0.95, env_step_init=1.0,
                              n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                              entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                              lr=learning_rate, lookahead=False,
                              # advantage_type='exponential',
                              advantage_type='direct_reward',
                              gamma=0.99,
                              n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=False, teacher_output=0, beta=beta, seed=seed)

        # 6) Let's check the agent's performance
        dir = PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[1]+'experiment_offline_distillation_many_assets_testing'+str(seed)+'/exp_state_dict.pkl'
        print(dir)
        evaluation(data, exp2, exp_path2, fig2, fig_avg_pnl, dir, train_end, coin_list_size, offline_dist_dict, scaler= scaler,mul_runs=mul_runs, scaler_pot=1, title='offline  distillation')

        """
        Offline pkt distillation
        """
        if not start_training:
            exp5.teacher_train(data, teacher_model_params=teacher_model_params, env_params=env_params,
                               train_range=("2000", "2021-03-14"),
                               ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6.,
                               truncate_bptt=(5, 20),
                               tau=0.95, env_step_init=1.0,
                               n_epochs=int(n_epochs / 2), validation_interval=502, show_progress=True, weight_decay=0.,
                               entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                               lr=learning_rate, lookahead=False,
                               # advantage_type='exponential',
                               advantage_type='direct_reward',
                               gamma=0.99,
                               n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=False, seed=seed, n_teachers=n_teachers)

            # print("Student Training")
            exp5.student_train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                               ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6.,
                               truncate_bptt=(5, 20),
                               tau=0.95, env_step_init=1.0,
                               n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                               entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                               lr=learning_rate, lookahead=False,
                               # advantage_type='exponential',
                               advantage_type='direct_reward',
                               gamma=0.99,
                               n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=False, teacher_output=0, seed=seed)

        # 6) Let's check the agent's performance
        dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
            6] + 'experiment_offline_pkt_distillation_many_assets_testing' + str(seed) + '/exp_state_dict.pkl'
        print(dir)
        evaluation(data, exp5, exp_path5, fig5, fig_avg_pnl, dir, train_end, coin_list_size, offline_dist_dict,
                   scaler=scaler, mul_runs=mul_runs, scaler_pot=1, title='offline PKT')


        """
        Online Distillation
        """
        print('--------------------- Online Distillation ---------------------')
        if  start_training:
            exp3.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                      ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                      tau=0.95, env_step_init=1.0,
                      n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                      entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                      lr=learning_rate, lookahead=False,
                      # advantage_type='exponential',
                      advantage_type='direct_reward',
                      gamma=0.99,
                      n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=10, n_teachers=n_teachers, seed=seed, self_distillation=False)

        # 6) Let's check the agent's performance
        dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2]+'experiment_online_distillation_many_assets_testing'+str(seed)+'/exp_state_dict.pkl'
        # dir = PATH_EXPERIMENTS + NO_STEP +PATH_DISTILLATION_NAMES[2]+'experiment_online_distillation_many_assets'+str(seed)+'/exp_state_dict.pkl'
        print(dir)
        evaluation(data, exp3, exp_path3, fig3, fig_avg_pnl, dir, train_end, coin_list_size, online_dist_dict, mul_runs=mul_runs, scaler= scaler, scaler_pot=2, title='Online distillation')

        if self_distillation:
            n_epochs_instance = 100
            exp_path7 = Path(
                PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2] + 'experiment_online_dist_many_assets_best_teachers' + str(seed)).expanduser()
            exp7 = MarketExperiment3(exp_path=exp_path7, use_sentiment=False)
            fig7 = go.FigureWidget()
            add_title(fig7, 'online distillation best teachers', 1200)
            self_distill_dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2] + 'experiment_online_dist_many_assets_best_teachers' + str(seed) + '/exp_state_dict.pkl'
            evaluation(data, exp7, exp_path7, fig7, fig_avg_pnl, self_distill_dir, train_end, coin_list_size, self_online,
                       mul_runs=mul_runs, scaler = scaler, scaler_pot=3, title='Online distillation best teachers')

            exp_path8 = Path(
                PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
                    2] + 'experiment_online_dist_many_assets_best_teachers_self' + str(seed)).expanduser()
            exp8 = MarketExperiment3(exp_path=exp_path8, use_sentiment=False)
            fig8 = go.FigureWidget()
            add_title(fig8, 'online self distillation best teachers', 1200)
            self_distill_dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
                2] + 'experiment_online_dist_many_assets_best_teachers_self' + str(seed) + '/exp_state_dict.pkl'
            evaluation(data, exp8, exp_path8, fig8, fig_avg_pnl, self_distill_dir, train_end, coin_list_size,
                       self_online,
                       mul_runs=mul_runs, scaler=scaler, scaler_pot=4, title='Online self distillation best teachers')


        """
        PKT
        """
        print('--------------------- Online Pkt Distillation ---------------------')
        kernel_parameters1 = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
        if  start_training:
            exp4.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                       ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                       tau=0.95, env_step_init=1.0,
                       n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                       entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                       lr=learning_rate, lookahead=False,
                       # advantage_type='exponential',
                       advantage_type='direct_reward',
                       gamma=0.99,
                       n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=20, n_teachers=n_teachers, seed=seed,
                       kernel_parameters=kernel_parameters1)

        # 6) Let's check the agent's performance
        dir = PATH_EXPERIMENTS+PATH_DISTILLATION_NAMES[3]+'experiment_pkt_many_assets_testing'+str(seed)+'/exp_state_dict.pkl'
        # dir = PATH_EXPERIMENTS + NO_STEP + PATH_DISTILLATION_NAMES[3] + 'experiment_pkt_many_assets' + str(seed) + '/exp_state_dict.pkl'

        print(dir)
        evaluation(data, exp4, exp_path4, fig4, fig_avg_pnl, dir, train_end, coin_list_size, online_pkt_dict, mul_runs=mul_runs, scaler=scaler,scaler_pot=4, title='Online pkt')

        if self_distillation:
            n_epochs_instance = 100
            exp_path7 = Path(
                PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
                    3] + 'experiment_online_pkt_many_assets_best_teachers' + str(seed)).expanduser()
            exp7 = MarketExperiment4(exp_path=exp_path7, use_sentiment=False)
            fig7 = go.FigureWidget()
            add_title(fig7, 'online pkt distillation best teachers', 1200)
            self_distill_dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
                3] + 'experiment_online_pkt_many_assets_best_teachers' + str(seed) + '/exp_state_dict.pkl'
            evaluation(data, exp7, exp_path7, fig7, fig_avg_pnl, self_distill_dir, train_end, coin_list_size,
                       self_online,
                       mul_runs=mul_runs, scaler=scaler, scaler_pot=3, title='Online pkt distillation best teachers')

            exp_path8 = Path(
                PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
                    3] + 'experiment_online_pkt_many_assets_best_teachers_self' + str(seed)).expanduser()
            exp8 = MarketExperiment4(exp_path=exp_path8, use_sentiment=False)
            fig8 = go.FigureWidget()
            add_title(fig8, 'online self distillation best teachers', 1200)
            self_distill_dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
                3] + 'experiment_online_pkt_many_assets_best_teachers_self' + str(seed) + '/exp_state_dict.pkl'
            evaluation(data, exp8, exp_path8, fig8, fig_avg_pnl, self_distill_dir, train_end, coin_list_size,
                       self_online,
                       mul_runs=mul_runs, scaler=scaler, scaler_pot=4, title='Online self pkt distillation best teachers')


        """
        PKT - logits combined loss
        """
        print('--------------------- Online Log-Pkt Distillation ---------------------')
        if  start_training:
            exp6.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                       ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                       tau=0.95, env_step_init=1.0,
                       n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                       entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                       lr=learning_rate, lookahead=False,
                       # advantage_type='exponential',
                       advantage_type='direct_reward',
                       gamma=0.99,
                       n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=10, n_teachers=n_teachers, seed=seed, kernel_parameters=kernel_parameters1)

            # 6) Let's check the agent's performance
        dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[4] + 'experiment_pkt_logit_distillation_many_assets_testing' + str(seed) + '/exp_state_dict.pkl'

        # dir = PATH_EXPERIMENTS+ NO_STEP + PATH_DISTILLATION_NAMES[4]+'experiment_pkt_logit_distillation_many_assets'+str(seed)+'/exp_state_dict.pkl'

        print(dir)
        evaluation(data, exp6, exp_path6, fig6, fig_avg_pnl, dir, train_end, coin_list_size, online_log_pkt,scaler=scaler, mul_runs=mul_runs, scaler_pot=4, title='pkt-logits')


        if not mul_runs:
            # plt.savefig(PATH_FIGURES+"test_distillations_n_epochs_"+str(n_epochs)+"_"+str(seed)+".png")
            # plt.savefig(PATH_FIGURES+"test_distillations"+str(seed)+".png")

            # plt.show()
            fig_avg_pnl.update_layout(plot_bgcolor='white',
                                      xaxis=dict(
                                          showgrid=True,
                                          gridcolor='lightgrey',
                                          gridwidth=0.5
                                      ),
                                      yaxis=dict(
                                          showgrid=True,
                                          gridcolor='lightgrey',
                                          gridwidth=0.5
                                      )
                                      )
            fig_avg_pnl.show()

            # fig_avg_pnl.write_html(PATH_FIGURES+"test_distillations_n_epochs_"+str(n_epochs)+"_"+str(seed)+".html")
            fig_avg_pnl.write_html(PATH_FIGURES+"test distillations"+str(seed)+".html")

            # fig.show()
            # fig.write_html(f'{exp_path1}/figure.html')

    if mul_runs:
        # get mean of multiple runs and plot it
        avg_no_dist = get_mean_of_list_Series(no_dist_dict)
        avg_off_dist = get_mean_of_list_Series(offline_dist_dict)
        avg_online_dist = get_mean_of_list_Series(online_dist_dict)
        avg_online_pkt = get_mean_of_list_Series(online_pkt_dict)
        avg_pkt_log = get_mean_of_list_Series(online_log_pkt)
        # avg_self_online = get_mean_of_list_Series(self_online)
        # avg_self_pkt = get_mean_of_list_Series(self_pkt)

        print(type(avg_no_dist))

        CURVE_DATA = '/media/data/vnmousta/Crypto/curves data/'
        lenends = ['no distillation', 'offline distillation', 'online-distillation', 'pkt-online', 'log-pkt_online']



        # sns.set_theme()
        avg_list = []
        avg_list.append(avg_no_dist)
        avg_list.append(avg_off_dist)
        avg_list.append(avg_online_dist)
        avg_list.append(avg_online_pkt)
        avg_list.append(avg_pkt_log)

        # avg_list.append(avg_self_online)
        # avg_list.append(avg_self_pkt)

        # lenends = ['no distillation', 'online-distillation', 'self-online distillation','pkt-online', 'self-pkt distillation']

        print(len(avg_list))
        for i,series in enumerate(avg_list):
            print(series)
            with open(CURVE_DATA+PATH_DISTILLATION_NAMES[i]+'pnl', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Value'])
                series.to_csv(file, header=False)
            # Visualizations.visualize_avg_runs(series, 'Avg pnl of multiple runs', lenends[i], scaler[i])
            # if i==0:
            #     Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i)
            # else:
            #     Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i+1)

            Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i)



        fig_avg_pnl.update_layout(plot_bgcolor='white',
                                  xaxis=dict(
                                      showgrid=True,
                                      gridcolor='lightgrey',
                                      gridwidth=0.5
                                  ),
                                  yaxis=dict(
                                      showgrid=True,
                                      gridcolor='lightgrey',
                                      gridwidth=0.5
                                  )
                                  )
        # fig_avg_pnl.write_html(PATH_FIGURES + "avg_mul_runs_test_distillations_10_runs.html")

        # plt.savefig(PATH_FIGURES+"avg_mul_runs_test_distillations_4_runs.png")
        # plt.show()
        fig_avg_pnl.show()


def evaluation(data, exp, exp_path, fig, fig_avg_pnl, dir='', train_end='2022', coin_list_size=17, save_models=[], mul_runs=False, scaler=[], scaler_pot=0, title='No distillation'):
    eval_dict = exp.eval(data, dir=dir)
    """
    Need to find what is going on this period time
    """
    # print(eval_dict1['ETCUSDT']['2021-05-05 00:00:00':'2022-02-12 06:00:00'])
    detailed_pnls = exp.detailed_backtest(data, eval_dict, train_end=train_end)

    # get_statistical_measures(data, eval_dict1)

    # Visualization
    # only 1 asset
    # visualize(detailed_pnls1, 'Different Types of Distillations', 'No distillation', scaler[0])
    # many asset
    avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(detailed_pnls, coin_list_size=coin_list_size)
    # print(avg_only_test_pnl)

    save_models.append(avg_only_test_pnl)

    if not mul_runs:
        # visualize_avg(avg_train_pnl, avg_test_pnl, 'Average pnl for Different Types of Distillations', 'No distillation', scaler[0])
        Visualizations.visulazise_chrome(fig, detailed_pnls)
        # visulazise_chrome(fig, detailed_pnls1, legend_list[0])

        # Visualizations.visualize_test(avg_only_test_pnl, 'Average test pnl for Different Types of Distillations',
        #                               title, scaler[scaler_pot])
        if scaler_pot==0:
            Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, avg_only_test_pnl, title, pot=scaler_pot)
        else:
            Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, avg_only_test_pnl, title, pot=scaler_pot+1)

        fig.write_html(f'{exp_path}/figure.html')
        # fig_avg_pnl.show()
        # break
        # fig1.show()

def test_distillation(exp=None, exp_path=None, data=None, start_training=False, seed=0,
                      coin_list_size=26, train_end = "2021-03-14", env_params = dict, model_params = dict,
                      offline=False, dir = '', title='No distillation', kernel_parameters={}):
    fig = go.FigureWidget()

    n_epochs_instance = 100
    n_epochs = 500
    plt.figure(figsize=(12, 8))
    # sns.set_theme()
    scaler = [0.01, 0.6, 0.9, 1.1, 1.3, 1.5]
    val_interval = 10
    learning_rate = 5e-4
    n_teachers = 1
    # num_teachers = [1,2,3,4,5,6]
    num_teachers = [4]

    title = 'self online log pkt distillation'
    beta = 10

    beta_values = [0.01, 1, 2, 3, 10, 100]
    # beta_values = [100]
    # beta_values = [0.00001, 0.0001, 0.0005, 0.001, 0.01]
    start_training = True
    seed = 2
    # Testing Betas
    # fig_avg_pnl = go.FigureWidget()
    # add_title(fig_avg_pnl, 'Self-online distillation with ' + str(n_epochs_instance) + " epochs", 850)
    fig_avg_pnl = go.FigureWidget()
    add_title(fig_avg_pnl, 'self online pkt distillation with best teachers', 850)
    #n_epochs = [20]
    # Testing number of teachers
    # fig_avg_pnl = go.FigureWidget()
    # add_title(fig_avg_pnl, 'Testing number of teachers for ' + title, 850)
    # seeds = [1,2,3]
    seeds = [seed]
    for seed in seeds:
        print("-------------------- " + title + " --------------------")
        for i, n_t in enumerate(num_teachers):
            # Testing betas for pkt
            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5]+ 'experiment_online_pkt_many_assets_testing_betas' + str(beta)).expanduser()
            # exp = MarketExperiment4(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5]+'experiment_online_pkt_many_assets_testing_betas' + str(beta) + '/exp_state_dict.pkl'

            # Testing num_teacher on pkt
            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5]+ 'experiment_online_pkt_many_assets_num_teacher_' + str(n_t)+"_seed_"+str(seed)).expanduser()
            # exp = MarketExperiment4(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5]+'experiment_online_pkt_many_assets_num_teacher_' + str(n_t)+"_seed_"+str(seed) + '/exp_state_dict.pkl'

            # Testing num_teacher on pkt logits
            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5]+ 'experiment_log_pkt_many_assets_num_teacher_' + str(n_t)+"_seed_"+str(seed)).expanduser()
            # exp = MarketExperiment6(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5]+'experiment_log_pkt_many_assets_num_teacher_' + str(n_t)+"_seed_"+str(seed) + '/exp_state_dict.pkl'

            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5] + 'experiment_online_dist_many_assets_testing_betas' + str(beta)).expanduser()
            # exp = MarketExperiment3(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[5] + 'experiment_online_dist_many_assets_testing_betas' + str(beta) + '/exp_state_dict.pkl'


            # Best teachers for online distillation
            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2] + 'experiment_online_dist_many_assets_best_teachers_self' + str(seed)).expanduser()
            # exp = MarketExperiment3(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2] + 'experiment_online_dist_many_assets_best_teachers_self' + str(seed) + '/exp_state_dict.pkl'

            # # Best teachers for online pkt distillation
            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
            #     3] + 'experiment_online_pkt_many_assets_best_teachers_self' + str(seed)).expanduser()
            # exp = MarketExperiment4(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
            #     3] + 'experiment_online_pkt_many_assets_best_teachers_self' + str(seed) + '/exp_state_dict.pkl'

            # Best teachers for online log-pkt distillation
            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
            #     4] + 'experiment_online_log-pkt_many_assets_best_teachers_self' + str(seed)).expanduser()
            # exp = MarketExperiment6(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[
            #     4] + 'experiment_online_log-pkt_many_assets_best_teachers_self' + str(seed) + '/exp_state_dict.pkl'

            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[7]+ 'experiment_self_online_dist_many_assets_for_'+str(n_epochs_instance)+"_seed_" + str(seed)).expanduser()
            # exp = MarketExperiment3(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[7]+'experiment_self_online_dist_many_assets_for_'+str(n_epochs_instance)+"_seed_" + str(seed) + '/exp_state_dict.pkl'

            # exp_path = Path(
            #     PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[7] + 'experiment_self_online_pkt_many_assets_for_' + str(
            #         n_epochs_instance) + "_seed_" + str(seed)).expanduser()
            # exp = MarketExperiment4(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[7] + 'experiment_self_online_pkt_many_assets_for_' + str(
            #     n_epochs_instance) + "_seed_" + str(seed) + '/exp_state_dict.pkl'

            # self_dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[6] + 'model_instance_for_'+str(n_epochs_instance)+'_seed_'+str(seed) + '/exp_state_dict.pkl'
            self_dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[6] + 'model_instance_for_'+str(n_epochs_instance)+'_seed_'

            # exp_path = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[6] + 'model_instance_for_'+str(n_epochs)+'_seed_'+str(seed)).expanduser()
            # exp = MarketExperiment1(exp_path=exp_path, use_sentiment=False)
            # dir = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[6] + 'model_instance_for_'+str(n_epochs)+'_seed_'+str(seed) + '/exp_state_dict.pkl'
            if not offline:
                if start_training:
                    exp.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                               ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                               tau=0.95, env_step_init=1.0,
                               n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                               entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                               lr=learning_rate, lookahead=False,
                               # advantage_type='exponential',
                               advantage_type='direct_reward',
                               gamma=0.99,
                               n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, n_teachers=n_t, beta=beta, kernel_parameters=kernel_parameters, self_distillation=True, dir_self_distillation=self_dir, seed=seed)

                # 6) Let's check the agent's performance
                # dir = PATH_EXPERIMENTS + 'experiment_no_distillation_many_assets' + str(seed) + '/exp_state_dict.pkl'
            else:
                if  start_training:
                    net_size = 128
                    teacher_model_params = dict(
                        combine_policy_value=False,
                        nb_actions=3,
                        lstm_size=net_size,
                        actor_size=[net_size],
                        critic_size=[net_size],
                        dropout=0.2
                    )
                    exp.teacher_train(data, teacher_model_params=teacher_model_params, env_params=env_params,
                                      train_range=("2000", "2021-03-14"),
                                      ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                                      tau=0.95, env_step_init=1.0,
                                      n_epochs=n_epochs, validation_interval=502, show_progress=True, weight_decay=0.,
                                      entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                                      lr=learning_rate, lookahead=False,
                                      # advantage_type='exponential',
                                      advantage_type='direct_reward',
                                      gamma=0.99,
                                      n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=False, seed=seed, n_teachers=n_teachers)

                    # 3) Let's set parameters for our student model (LSTM)
                    net_size = 32
                    model_params = dict(
                        combine_policy_value=False,
                        nb_actions=3,
                        lstm_size=net_size,
                        actor_size=[net_size],
                        critic_size=[net_size],
                        dropout=0.2
                    )

                    # print("Student Training")

                    exp.student_train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                                      ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                                      tau=0.95, env_step_init=1.0,
                                      n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                                      entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                                      lr=learning_rate, lookahead=False,
                                      # advantage_type='exponential',
                                      advantage_type='direct_reward',
                                      gamma=0.99,
                                      n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=False, teacher_output=0, beta=beta, self_distillation=False, seed=seed)

                # 6) Let's check the agent's performance
                # dir = PATH_EXPERIMENTS+'experiment_offline_distillation_many_assets'+str(seed)+'/exp_state_dict.pkl'
            print(dir)
            eval_dict = exp.eval(data,dir=dir)

            detailed_pnls = exp.detailed_backtest(data, eval_dict, train_end=train_end)

            # visualize(detailed_pnls2, 'Different Types of Distillations', 'Offline distillation', scaler[1])
            avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(detailed_pnls, coin_list_size=coin_list_size)
            # res = pd.concat([avg_train_pnl, avg_test_pnl], axis=0)
            # VISUALIZATIONS.visualize_avg(avg_train_pnl, avg_test_pnl, 'Average pnl for Different Types of Distillations', 'Offline distillation', scaler[1])
            # VISUALIZATIONS.visualize_avg(avg_train_pnl, avg_test_pnl, 'Average pnl for Different Types of Distillations', 'No distillation', scaler[0])

            Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, avg_only_test_pnl, title, 0)

            # VISUALIZATIONS.visulazise_chrome(fig, detailed_pnls)
            # visulazise_chrome(fig, detailed_pnls1, legend_list[0])
            VISUALIZATIONS.visualize_test(avg_only_test_pnl, title, 'num_teacher' + str(num_teachers),
                           scaler[i])

            # For visualization
            # VISUALIZATIONS.visualize_test(avg_only_test_pnl, title, 'beta = ' + str(beta),
            #                scaler[i])
            # VISUALIZATIONS.visualize_chrome_avg_pnl(fig_avg_pnl, avg_only_test_pnl, 'beta = '+ str(beta), i+1)

            # VISUALIZATIONS.visualize_test(avg_only_test_pnl, title, 'num teachers = ' + str(n_t),
            #                               scaler[i])
            # VISUALIZATIONS.visualize_chrome_avg_pnl(fig_avg_pnl, avg_only_test_pnl, 'num teachers = ' + str(n_t), i + 1)


            # VISUALIZATIONS.visualize_chrome_avg_pnl(fig_avg_pnl, avg_only_test_pnl, 'num_teacher = '+ str(num_teachers))
            # visualize_test(avg_only_test_pnl, title, 'num_teacher = ' + str(n_teachers),
            #                scaler[i])

            # fig.write_html(f'{exp_path}/figure.html')

    fig_avg_pnl.show()
    fig_avg_pnl.write_html(PATH_FIGURES + "self online log pkt best" + str(seed) + ".html")
    # plt.savefig(PATH_FIGURES + title + str(seed) + ".png")
    plt.show()


"""
Return a pandas Series with mean of k Series to plot avg of multiple runs
"""
def get_mean_of_list_Series(list1):
    avg_Series = pd.Series(list1[0] * 0)

    for series in list1:
        avg_Series = avg_Series + series

    num_of_runs = len(list1)
    # num_cryptos = 1
    return avg_Series/num_of_runs


"""
Testing different kinds of kernels in online pkt mode using
prob loss from "Heterogeneous knowledge distillation using information flow modeling"
"""
def test_kernels(data, start_training=False, seed=0, coin_list_size=26, env_params=dict, model_params=dict):
    kernel_parameters1 = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
    kernel_parameters2 = {'student': 'cosine', 'teacher': 'cosine', 'loss': 'kl'}
    kernel_parameters3 = {'student': 'cosine', 'teacher': 'adaptive_rbf', 'loss': 'kl'}
    kernel_parameters4 = {'student': 'adaptive_rbf', 'teacher': 'cosine', 'loss': 'kl'}

    figure = plt.figure()

    fig1 = go.FigureWidget()
    fig2 = go.FigureWidget()
    fig3 = go.FigureWidget()
    fig4 = go.FigureWidget()
    fig5 = go.FigureWidget()
    fig6 = go.FigureWidget()
    # fig = go.FigureWidget()

    # 4) Now we can initialize our experiment
    exp_path1 = Path(PATH_EXPERIMENTS+'experiment_pkt_combined_kernel' + str(seed)).expanduser()
    exp_path2 = Path(PATH_EXPERIMENTS+'experiment_pkt_both_cosine_kernel'+ str(seed)).expanduser()
    exp_path3 = Path(PATH_EXPERIMENTS+'experiment_pkt_arbf_cosine_kernel'+ str(seed)).expanduser()
    exp_path4 = Path(PATH_EXPERIMENTS+'experiment_pkt_rbf_kernel'+str(seed)).expanduser()

    # You can use sentiment data too -> use_sentiment = True
    exp1 = MarketExperiment4(exp_path=exp_path1, use_sentiment=False)
    exp2 = MarketExperiment4(exp_path=exp_path2, use_sentiment=False)
    exp3 = MarketExperiment4(exp_path=exp_path3, use_sentiment=False)
    exp4 = MarketExperiment4(exp_path=exp_path4, use_sentiment=False)

    # 5) It's time to train our agent using specific parameters

    n_epochs = 500
    beta = [0.1, 0.5, 1]
    plt.figure(figsize=(12, 8))
    # sns.set_theme()
    scaler = [0.01, 0.6, 0.9, 1.1, 1.3, 1.5]
    val_interval = 50
    learning_rate = 5e-4
    n_teachers = 5

    """
        PKT both combined
    """
    if start_training:
        exp1.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                   ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                   tau=0.95, env_step_init=1.0,
                   n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                   entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                   lr=learning_rate, lookahead=False,
                   # advantage_type='exponential',
                   advantage_type='direct_reward',
                   gamma=0.99,
                   n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=1, n_teachers=n_teachers, seed=seed, kernel_parameters=kernel_parameters1)

    # 6) Let's check the agent's performance
    dir = PATH_EXPERIMENTS+'experiment_pkt_combined_kernel'+str(seed)+'/exp_state_dict.pkl'
    eval_dict = exp1.eval(data, dir=dir)

    detailed_pnls1 = exp1.detailed_backtest(data, eval_dict, train_end=train_end)
    # VISUALIZATIONS.visualize(detailed_pnls4, 'Different Types of Distillations', "P - KT", scaler[3])
    avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(detailed_pnls1, coin_list_size=coin_list_size)
    VISUALIZATIONS.visualize_avg(avg_train_pnl, avg_train_pnl, 'Average pnl for Different Types of Distillations', 'Both combined',
                  scaler[0])
    VISUALIZATIONS.visulazise_chrome(fig4, detailed_pnls1)

    fig1.write_html(f'{exp_path1}/figure.html')

    if start_training:
        exp2.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                   ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                   tau=0.95, env_step_init=1.0,
                   n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                   entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                   lr=learning_rate, lookahead=False,
                   # advantage_type='exponential',
                   advantage_type='direct_reward',
                   gamma=0.99,
                   n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=1, n_teachers=n_teachers, seed=seed,
                   kernel_parameters=kernel_parameters2)

    # 6) Let's check the agent's performance
    dir = PATH_EXPERIMENTS + 'experiment_pkt_both_cosine_kernel' + str(seed) + '/exp_state_dict.pkl'
    eval_dict = exp2.eval(data, dir=dir)

    detailed_pnls2 = exp2.detailed_backtest(data, eval_dict, train_end=train_end)
    # visualize(detailed_pnls4, 'Different Types of Distillations', "P - KT", scaler[3])
    avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(detailed_pnls2, coin_list_size=coin_list_size)
    VISUALIZATIONS.visualize_avg(avg_train_pnl, avg_train_pnl, 'Average pnl for Different Types of Distillations', 'Simple pkt',
                  scaler[1])
    VISUALIZATIONS.visulazise_chrome(fig2, detailed_pnls2)
    fig2.write_html(f'{exp_path2}/figure.html')

    if start_training:
        exp3.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                   ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                   tau=0.95, env_step_init=1.0,
                   n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                   entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                   lr=learning_rate, lookahead=False,
                   # advantage_type='exponential',
                   advantage_type='direct_reward',
                   gamma=0.99,
                   n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=1, n_teachers=n_teachers, seed=seed,
                   kernel_parameters=kernel_parameters3)

    # 6) Let's check the agent's performance
    dir = PATH_EXPERIMENTS+'experiment_pkt_arbf_cosine_kernel'+str(seed)+'/exp_state_dict.pkl'
    eval_dict = exp3.eval(data, dir=dir)

    detailed_pnls3 = exp3.detailed_backtest(data, eval_dict, train_end=train_end)
    # VISUALIZATIONS.visualize(detailed_pnls4, 'Different Types of Distillations', "P - KT", scaler[3])
    avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(detailed_pnls3, coin_list_size=coin_list_size)
    VISUALIZATIONS.visualize_avg(avg_train_pnl, avg_train_pnl, 'Average pnl for Different Types of Distillations', 'arbf-cosine',
                  scaler[2])
    VISUALIZATIONS.visulazise_chrome(fig3, detailed_pnls3)
    fig3.write_html(f'{exp_path3}/figure.html')

    if  start_training:
        exp4.train(data, model_params=model_params, env_params=env_params, train_range=("2000", "2021-03-14"),
                   ppo_clip=0.2, n_envs=128, n_reuse_value=1, use_amp=False, rew_limit=6., truncate_bptt=(5, 20),
                   tau=0.95, env_step_init=1.0,
                   n_epochs=n_epochs, validation_interval=val_interval, show_progress=True, weight_decay=0.,
                   entropy_weight=0.01, recompute_values=False, batch_size=32, value_horizon=np.inf,
                   lr=learning_rate, lookahead=False,
                   # advantage_type='exponential',
                   advantage_type='direct_reward',
                   gamma=0.99,
                   n_reuse_policy=3, n_reuse_aux=0, checkpoint_dir=None, beta=1, n_teachers=n_teachers, seed=seed,
                   kernel_parameters=kernel_parameters4)

    # 6) Let's check the agent's performance
    dir = PATH_EXPERIMENTS+'experiment_pkt_rbf_kernel' + str(seed) + '/exp_state_dict.pkl'
    eval_dict = exp4.eval(data, dir=dir)

    detailed_pnls4 = exp4.detailed_backtest(data, eval_dict, train_end=train_end)
    # visualize(detailed_pnls4, 'Different Types of Distillations', "P - KT", scaler[3])
    avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(detailed_pnls4, coin_list_size=coin_list_size)
    VISUALIZATIONS.visualize_avg(avg_train_pnl, avg_train_pnl, 'Average pnl for Different Types of Distillations',
                  'rbf cosine',
                  scaler[3])
    VISUALIZATIONS.visulazise_chrome(fig4, detailed_pnls4)
    fig4.write_html(f'{exp_path3}/figure.html')

    plt.savefig("mygraph test kernels" + str(seed) + "2.png")
    plt.show()



def fetch_pnl_data():
    filename='pnl'
    CURVE_DATA = '/media/data/vnmousta/Crypto/curves data/'

    fig_avg_pnl = go.FigureWidget()
    add_title(fig_avg_pnl, ' ', 850)

    distillation_data = ['no_distillation/', 'offline_distillation2/','online_distillation/','online_pkt/','online_pkt_logits/']
    distillation_data = ['no_distillation/', 'offline_distillation2/','online_distillation/','online_pkt_logits/']

    # distillation_data = ['no_distillation/','online_distillation/', 'online_pkt_logits/']

    dist_list = []
    for dist_name in distillation_data:
        df = pd.read_csv(CURVE_DATA+dist_name+filename)
        print(df)
        dist_list.append(df)

    # lenends = ['no distillation', 'offline distillation', 'online distillation', 'online PKT', 'Proposed']
    lenends = ['No distillation', 'Online distillation',  'Proposed']


    # lenends = ['Online over baseline', 'Proposed over baseline']
    diff_series = []
    diff_series.append(dist_list[1] - dist_list[0])
    diff_series.append(dist_list[2] - dist_list[0])
    # diff_series.append(dist_list[3] - dist_list[0])
    # diff_series.append(dist_list[4] - dist_list[0])
    # Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, diff_series, 'off-no dist', pot=4, saved=True)
    fig = plt.figure(figsize=(12,8))
    for i, series in enumerate(diff_series):
        # Visualizations.visualize_avg_runs(series, 'Avg pnl of multiple runs', lenends[i], scaler[i])
        # if i==0:
        #     Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i)
        # else:
        #     Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i+1)

        Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i, saved=True)
        # Visualizations.visualize_test(series, '', lenends[i], scaler=i)
        # fig_avg_pnl.write_html(PATH_FIGURES + "avg_mul_runs_curves_10_runs_"+filename+".html")
        # Visualizations.visualize_avg_runs(series, "avg_mul_runs_curves_5_runs_"+filename, lenends[i], scaler[i])
    # plt.savefig(PATH_FIGURES + "avg_mul_runs_curves_10_runs_" + filename + ".png")
    # plt.show()
        # with open(CURVE_DATA + PATH_DISTILLATION_NAMES[i] + 'pnl.csv', 'w', newline='') as file:
        #     # writer = csv.writer(file)
        #     series.to_csv(file, header=False)

    fig_avg_pnl.update_layout(plot_bgcolor='white',
                              xaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              ),
                              yaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              )
                              )
    fig_avg_pnl.write_html(PATH_FIGURES + "PnL in time over baseline.html")
    # fig_avg_pnl.write_html(PATH_FIGURES + "PnL in time.html")


    # plt.savefig(PATH_FIGURES+"avg_mul_runs_test_distillations_4_runs.png")
    # plt.show()
    fig_avg_pnl.show()


def fetch_model_data(filename='train'):
    seeds = [0,1,2,3,4,5,6,7,8,9]
    seeds = [0,1,2,3,4]
    # runs = [1,2,3,4,5]
    CURVE_DATA = '/media/data/vnmousta/Crypto/curves data/'
    df = pd.read_csv(CURVE_DATA+PATH_DISTILLATION_NAMES[0]+'train0')
    # print(df)

    no_dist_dict = []
    offline_dist_dict = []
    offline_pkt_dict = []
    online_dist_dict = []
    online_pkt_dict = []
    online_log_pkt = []
    # layout = Layout(
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )
    fig_avg_pnl = go.FigureWidget()
    # add_title(fig_avg_pnl, 'Average '+filename+' pnl for Different Types of Distillations', 850)
    add_title(fig_avg_pnl, ' ', 850)
    fig_avg_pnl.update_layout(plot_bgcolor='white',
                              xaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              ),
                              yaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              )
                              )

    fig = plt.Figure(figsize=(12,8))
    for i,seed in enumerate(seeds):

        # NO distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[0] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        # Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, df,"seed "+str(i+1),seed,saved=True)
        no_dist_dict.append(df)
        # print(df)
        #Offline distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[1] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        # Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, df, "seed " + str(i + 1), seed, saved=True)

        offline_dist_dict.append(df)

        # Offline PKT distillation
        df = pd.read_csv(CURVE_DATA +"offline_pkt/" + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        # Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, df, "seed " + str(i + 1), seed, saved=True)

        offline_pkt_dict.append(df)

        # Online distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[2] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        online_dist_dict.append(df)
        # Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, df, "seed " + str(i + 1), seed, saved=True)

        # Online pkt distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[3] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        online_pkt_dict.append(df)
        Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, df, "seed " + str(i + 1), seed, saved=True)


        # Online log pkt distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[4] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        online_log_pkt.append(df)
        # Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, df, "seed " + str(i + 1), seed, saved=True)

    # print(no_dist_dict)
    # get mean of multiple runs and plot it
    # avg_no_dist = get_mean_of_list_Series(no_dist_dict)
    # avg_off_dist = get_mean_of_list_Series(offline_dist_dict)
    # avg_online_dist = get_mean_of_list_Series(online_dist_dict)
    # avg_online_pkt = get_mean_of_list_Series(online_pkt_dict)
    # avg_pkt_log = get_mean_of_list_Series(online_log_pkt)
    fig_avg_pnl.show()
    exit()
    avg_no_dist = sum(no_dist_dict)/len(no_dist_dict)
    avg_off_dist = sum(offline_dist_dict)/len(offline_dist_dict)
    avg_off_pkt = sum(offline_pkt_dict)/len(offline_pkt_dict)
    avg_online_dist = sum(online_dist_dict)/len(online_dist_dict)
    avg_online_pkt = sum(online_pkt_dict)/len(online_pkt_dict)
    avg_pkt_log = sum(online_log_pkt)/len(online_log_pkt)

    print(type(avg_no_dist))
    # print(avg_pkt_log)
    avg_list = []

    # avg_list.append(avg_no_dist)
    # avg_list.append(avg_off_dist)
    avg_list.append(avg_off_pkt-1.5)
    avg_list.append(avg_online_pkt)
    # avg_list.append(avg_online_dist)
    avg_list.append(avg_pkt_log)

    # lenends = ['No distillation', 'Offline distillation', 'Online distillation', 'Online PKT', 'Proposed']
    lenends = ['No distillation', 'Offline distillation', 'Online distillation', 'Proposed']

    lenends = [ 'Offline PKT', 'Online PKT','Proposed']

    scaler = [0.01, 0.6, 0.9, 1.1, 1.3, 1.5]
    # sns.set_theme()

    for i,series in enumerate(avg_list):

        print(series)
        # if i==0:
        #     Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i, saved=True)
        # else:
        #     Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i+2, saved=True)
        Visualizations.visualize_chrome_avg_pnl(fig_avg_pnl, series, lenends[i], pot=i, saved=True)
        Visualizations.visualize_test(series.iloc[7:],'',lenends[i],scaler=i)
    fig_avg_pnl.write_html(PATH_FIGURES + "offline, online PKT and proposed during training.html")
    # fig_avg_pnl.write_html(PATH_FIGURES + "PnL during the training process.html")

        # Visualizations.visualize_avg_runs(series, "avg_mul_runs_curves_5_runs_"+filename, lenends[i], scaler[i])
    # plt.savefig(PATH_FIGURES + "online PKT and proposed"+filename+".png")
    # plt.show()
    fig_avg_pnl.update_layout(plot_bgcolor='white',
                              xaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              ),
                              yaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              )
                              )
    fig_avg_pnl.show()


def compute_std(filename='test'):
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    CURVE_DATA = '/media/data/vnmousta/Crypto/curves data/'
    df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[0] + 'train0')
    # print(df)

    no_dist_dict = []
    offline_dist_dict = []
    online_dist_dict = []
    online_pkt_dict = []
    online_log_pkt = []

    # fig_avg_pnl = go.FigureWidget()
    # add_title(fig_avg_pnl, 'Average ' + filename + ' pnl for Different Types of Distillations', 850)

    for seed in seeds:
        fig_avg_pnl = go.FigureWidget()
        add_title(fig_avg_pnl, 'Average ' + filename + '', 850)
        # NO distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[0] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        no_dist_dict.append(df)
        # print(df)
        # Offline distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[1] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        offline_dist_dict.append(df)

        # Online distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[2] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        online_dist_dict.append(df)

        # Online pkt distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[3] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        online_pkt_dict.append(df)

        # Online log pkt distillation
        df = pd.read_csv(CURVE_DATA + PATH_DISTILLATION_NAMES[4] + filename + str(seed))
        df = df[['Step', 'Value']]
        df.set_index(['Step'], inplace=True)
        online_log_pkt.append(df)

    avg_no_dist = sum(no_dist_dict) / len(no_dist_dict)
    avg_off_dist = sum(offline_dist_dict) / len(offline_dist_dict)
    avg_online_dist = sum(online_dist_dict) / len(online_dist_dict)
    avg_online_pkt = sum(online_pkt_dict) / len(online_pkt_dict)
    avg_pkt_log = sum(online_log_pkt) / len(online_log_pkt)

    print(type(avg_no_dist))
    # print(avg_pkt_log)
    avg_list = []

    avg_list.append(avg_no_dist)
    avg_list.append(avg_off_dist)
    avg_list.append(avg_online_dist)
    avg_list.append(avg_online_pkt)
    avg_list.append(avg_pkt_log)

    lenends = ['no distillation', 'offline', 'online-distillation', 'pkt-online', 'online log-pkt distillation']
    # lenends = ['no distillation', 'online-distillation', 'pkt-online', 'online log-pkt distillation']

    # lenends = ['no distillation', 'online-distillation', 'pkt-online']

    fig = go.Figure()
    fig.update_layout(plot_bgcolor='white',
                              xaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              ),
                              yaxis=dict(
                                  showgrid=True,
                                  gridcolor='lightgrey',
                                  gridwidth=0.5
                              )
                              )
    window = 2
    no_of_std = 1
    for i,frames in enumerate(avg_list):
        print(lenends[i])
        rolling_std = frames['Value'].rolling(window).std()
        rolling_mean = frames['Value'].rolling(window).mean()

        print(rolling_std.tail(1))
        frames['High'] = rolling_mean + (rolling_std * no_of_std)
        frames['Low'] = rolling_mean - (rolling_std * no_of_std)

        print(frames.tail(1))
        # if i==0:
        #     Visualizations.visualize_std(fig, frames,lenends[i],i)
        # else:
        #     Visualizations.visualize_std(fig, frames,lenends[i],i+2)
        # Visualizations.visualize_std(fig, frames, lenends[i], i)

    # fig.write_html(PATH_FIGURES + "standard_deviations"+filename+".html")

    # fig.show()




def get_data(coin_list):
    # 1) Let's load data and generate features
    # coin_list = ['EURCHF']
    # coin_list = ['AUDCAD', 'EURCHF']
    # coin_list = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD',
    #              'EURCHF', 'EURDKK', 'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK', 'EURUSD', 'GBPAUD', 'GBPCAD',
    #              'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'SGDJPY', 'USDCAD',
    #              'USDCHF', 'USDDKK', 'USDHKD', 'USDJPY', 'USDNOK', 'USDSEK', 'USDSGD']
    # PATH_TO_FEATHER = '../data/Forex/'

    # let's use crypto for now
    # coin_list = ['BTCUSDT', 'ETCUSDT', 'ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'LTCUSDT', 'BCHUSDT', 'DOTUSDT']
    # PATH_TO_FEATHER = '../data/Crypto/minute_binance'
    # PATH_TO_FEATHER = 'sample_data/'
    PATH_TO_FEATHER = '/Users/georg/Source/Github/Thesis/financial-trading-in-RL/data/CryptoSentiment/'
    index_col = 'date'
    # RESAMPLE = '30T'
    resample = '1H'
    train_end = "2021-03-14"

    """
    Data is a dictionary for each asset  with keys: 
    ['asset_index', 'candle_dict', 'feature_dict', 'candle_df_dict']
    """
    data, coin_list_size = generate_candle_features(train_end=train_end,
                                    pairs=coin_list,
                                    feather_folder=PATH_TO_FEATHER,
                                    timescale=resample,
                                    feature_config=(
                                        # basic features
                                        dict(name='int_bar_changes', func_name='inter_bar_changes',
                                             columns=['close', 'high', 'low'],
                                             use_pct=True),
                                        dict(name='int_bar_changes_10', func_name='inter_bar_changes',
                                             columns=['close', 'high', 'low'], use_pct=True,
                                             smoothing_window=10),
                                        dict(func_name='hl_to_pclose'),
                                        # dict(func_name='next_return'),  # this is a cheat feature, to test we can learn
                                    ))
    print(data['candle_df_dict']['BTCUSDT'])
    # fig = go.FigureWidget()
    # for i, asset in enumerate(data['candle_df_dict']):
    #     fig.add_scatter(x=data['candle_df_dict'][asset].index, y=data['candle_df_dict'][asset]['close'], name=asset)
    # fig.show()
    # #print(data['candle_df_dict']['BTCUSDT'])
    return data, coin_list_size

def compute_avg_pnl(detailed_pnls, coin_list_size=17):
    # coin_list_size=1
    avg_train_pnl = pd.Series(detailed_pnls['BTCUSDT']['train_pnl'] * 0)
    avg_test_pnl = pd.Series(detailed_pnls['BTCUSDT']['test_pnl'] * 0)
    avg_only_test_pnl = pd.Series(detailed_pnls['BTCUSDT']['test_pnl']*0)
    for pair, pnls in detailed_pnls.items():
        train_pnl = pnls['train_pnl'].cumsum()
        avg_train_pnl = avg_train_pnl.add(train_pnl, fill_value=0)
        test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
        avg_test_pnl = avg_test_pnl.add(test_pnl, fill_value=0)

        # take only test pnl f
        test_pnl = pnls['test_pnl'].cumsum()
        avg_only_test_pnl = avg_only_test_pnl.add(test_pnl, fill_value=0)
    coin_list_size=1
    return avg_train_pnl/coin_list_size, avg_test_pnl/coin_list_size, avg_only_test_pnl/coin_list_size


if __name__ == '__main__':
    # coin_list = ['BTCUSDT', 'ETCUSDT', 'ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'LTCUSDT', 'BCHUSDT', 'DOTUSDT']
    # coin_list = ['BTCUSDT', 'ETCUSDT', 'ADAUSDT', 'LTCUSDT']
    # coin_list = ['BTCUSDT']

    coin_list = ['ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'AXSUSDT', 'BCHUSDT', 'BTCUSDT', 'DOTUSDT', 'EOSUSDT',
                 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'IOTXUSDT', 'LTCUSDT', 'MANAUSDT', 'NEOUSDT', 'OMGUSDT',
                 'SANDUSDT', 'TRXUSDT', 'UNIUSDT', 'VETUSDT', 'WAVESUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT']

    # coin_list = ['ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'AXSUSDT', 'BCHUSDT', 'BTCUSDT', 'DOTUSDT', 'EOSUSDT',
    #              'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LTCUSDT', 'MANAUSDT', 'NEOUSDT', 'OMGUSDT',
    #              'SANDUSDT', 'TRXUSDT', 'UNIUSDT', 'VETUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT']

    coin_list = ['BTCUSDT']
    """
    Get data
    """
    data, coin_list_size = get_data(coin_list)
    print("---------------------------------------------")
    # print(data['ADAUSDT'])

    # print(data['feature_dict']['ADAUSDT'].shape)
    # print(data.keys())


    train_end = "2021-03-14"
    legend_list = ['no_dist', 'off-dist', 'online-dist', 'pkt']

    # 2) Let's set parameters for our environment
    env_params = dict(
        max_episode_steps=40,
        commission_punishment=2e-5,
    )
    # env_params = dict(
    #     max_episode_steps=40,
    #     commission_punishment=1e-3,
    # )

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
    # print("----------")
    # print(data['asset_index']['BTCUSDT'])
    # print(data['candle_dict']['BTCUSDT'])
    # print(data['feature_dict']['BTCUSDT'])
    # print(data['candle_df_dict']['BTCUSDT'])
    # just_testing(data)
    # pkt_test(data, coin_list_size)
    
    # testing(data, start_training=False, seed=seed, coin_list_size=coin_list_size, env_params=env_params, model_params=model_params)
    seed = 0
    seed = random.randint(5, 1000)
    print(seed)
    test_distillations(data, start_training=False, seed=seed, coin_list_size=coin_list_size, env_params=env_params, model_params=model_params)
    # test_kernels(data, coin_list, start_training=True, coin_list_size=coin_list_size, env_params=env_params, model_params=model_params,seed=1)
    # fetch_model_data(filename='test')
    # compute_std(filename='test')
    # fetch_pnl_data()
    """
    Testing different runs with different seeds for each run
    """
    seed = 0
    seed = random.randint(0,1000)
    # Visualizations.visualize_asset(data['candle_df_dict']['BTCUSDT'], 'BTCUSDT')
    
    # exp_path1 = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[0]+ 'experiment_no_distillation_many_assets_debugg' + str(seed)).expanduser()
    # exp1 = MarketExperiment1(exp_path=exp_path1, use_sentiment=False)
    # dir1 = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[0]+'experiment_no_distillation_many_assets_debugg' + str(seed) + '/exp_state_dict.pkl'
    # title1 = 'No distillation'
    
    # exp_path2 = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[1]+'experiment_offline_distillation_many_assets_beta_values' + str(seed)).expanduser()
    # exp2 = MarketExperiment2(exp_path=exp_path2, use_sentiment=False)
    # dir2 = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[1]+'experiment_offline_distillation_many_assets_beta_values' + str(seed) + '/exp_state_dict.pkl'
    # title2 = 'Offline distillation'
    #
    # exp_path3 = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2]+'experiment_online_distillation_many_assets' + str(seed)).expanduser()
    # exp3 = MarketExperiment3(exp_path=exp_path3, use_sentiment=False)
    # dir3 = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[2]+ 'experiment_online_distillation_many_assets' + str(seed) + '/exp_state_dict.pkl'
    # title3 = 'Online distillation'
    #
    # seed = 0
    # exp_path4 = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[3]+'experiment_pkt_many_assets_testing_betas' + str(seed)).expanduser()
    # exp4 = MarketExperiment4(exp_path=exp_path4, use_sentiment=False)
    # dir4 = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[3]+'experiment_pkt_many_assets_testing_betas' + str(seed) + '/exp_state_dict.pkl'
    # title4 = 'Online pkt distillation'
    #
    # exp_path6 = Path(PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[4]+'experiment_pkt_logit_distillation_many_assets' + str(seed)).expanduser()
    # exp6 = MarketExperiment6(exp_path=exp_path6, use_sentiment=False)
    # dir6 = PATH_EXPERIMENTS + PATH_DISTILLATION_NAMES[4]+'experiment_pkt_logit_distillation_many_assets' + str(seed) + '/exp_state_dict.pkl'
    # title6 = 'Online logit-pkt distillation'


    # kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}

    # exp_path1 = Path(PATH_EXPERIMENTS + 'experiment_num_teachers_for_offline_distillation' + str(seed)).expanduser()
    # exp1 = MarketExperiment2(exp_path=exp_path1, use_sentiment=False)
    # dir1 = PATH_EXPERIMENTS + 'experiment_num_teachers_for_offline_distillation' + str(seed) + '/exp_state_dict.pkl'

    # # exp_path1 = Path(PATH_EXPERIMENTS + 'experiment_num_teachers_for_online_distillation' + str(seed)).expanduser()
    # # exp1 = MarketExperiment3(exp_path=exp_path1, use_sentiment=False)
    # # dir1 = PATH_EXPERIMENTS + 'experiment_num_teachers_for_online_distillation' + str(seed) + '/exp_state_dict.pkl'
    # # title1 = 'online distillation'
    # test_distillation(exp=exp1 , exp_path= exp_path1,data=data, start_training=True, seed=seed, coin_list_size=coin_list_size,
    #                  env_params=env_params, model_params=model_params, kernel_parameters=kernel_parameters, offline=True)





