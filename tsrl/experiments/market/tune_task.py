from tsrl.experiments.market.experiment_no_distillation import MarketExperiment

from tsrl.environments import generate_candle_features

import ray
from ray import tune


def run(config, data_dict=None, checkpoint_dir=None):
    # ray.tune.utils.wait_for_gpu(gpu_memory_limit=0.15, retry=20)
    exp_path = tune.get_trial_dir()
    exp = MarketExperiment(exp_path=exp_path)

    data = data_dict[config['interval']]
    # data = generate_candle_features(
    #     '2016', '1H',
    #     feature_config=(
    #         dict(name='int_bar_changes', func_name='inter_bar_changes',
    #              columns=['close', 'high', 'low'],
    #              use_pct=True),
    #         dict(name='int_bar_changes_50', func_name='inter_bar_changes',
    #              columns=['close', 'high', 'low'], use_pct=True,
    #              smoothing_window=50),
    #         dict(func_name='internal_bar_diff', use_pct=True),
    #         dict(func_name='hl_to_pclose'),
    #         dict(name='hlvol1', func_name='hl_volatilities',
    #              smoothing_window=10),
    #         dict(name='hlvol2', func_name='hl_volatilities',
    #              smoothing_window=50),
    #         dict(name='rvol1', func_name='return_volatilities',
    #              smoothing_window=10),
    #         dict(name='rvol2', func_name='return_volatilities',
    #              smoothing_window=50),
    #         dict(func_name='time_feature_day'),
    #         dict(func_name='time_feature_year'),
    #         dict(func_name='time_feature_month'),
    #         dict(func_name='time_feature_week')
    #     ))

    model_params = dict(
        nb_actions=3,
        **config['model_params']
    )
    exp.train(
        data, model_params=model_params, env_params=config['env_params'],
        checkpoint_dir=checkpoint_dir, show_progress=False,
        **config['train_params']
    )
    return


if __name__ == '__main__':
    ray.init(address='auto')
    data_dict = dict()
    for interval in ['1H']:
        data_dict[interval] = generate_candle_features(
            '2016', interval,
            feature_config=(
                dict(name='int_bar_changes', func_name='inter_bar_changes',
                     columns=['close', 'high', 'low'],
                     use_pct=True),
                dict(name='int_bar_changes_50', func_name='inter_bar_changes',
                     columns=['close', 'high', 'low'], use_pct=True,
                     smoothing_window=50),
                dict(func_name='internal_bar_diff', use_pct=True),
                dict(func_name='hl_to_pclose'),
                dict(name='hlvol1', func_name='hl_volatilities',
                     smoothing_window=10),
                dict(name='hlvol2', func_name='hl_volatilities',
                     smoothing_window=50),
                dict(name='rvol1', func_name='return_volatilities',
                     smoothing_window=10),
                dict(name='rvol2', func_name='return_volatilities',
                     smoothing_window=50),
                dict(func_name='time_feature_day'),
                dict(func_name='time_feature_year'),
                dict(func_name='time_feature_month'),
                dict(func_name='time_feature_week')
            ))

    config = dict(
        interval='1H',
        model_params=dict(
            dropout=0.,
            lstm_size=tune.choice([16, 32, 64, 128]),
            actor_size=tune.sample_from(lambda spec: [spec.config.model_params['lstm_size']]),
            critic_size=tune.sample_from(lambda spec: [spec.config.model_params['lstm_size']]),
        ),
        env_params=dict(
            max_episode_steps=100,
            commission_punishment=8e-5,
            static_limit=0.
        ),
        train_params=dict(
            n_epochs=5000,
            validation_interval=1000,
            rew_limit=tune.choice([1., 3., 6.]),
            lr=1e-3,
            advantage_type='direct_reward',
            weight_decay=tune.choice([1e-3, 1e-2, 0.]),
            entropy_weight=tune.choice([0.01, 0.001]),
            gamma=tune.choice([-0.1]),
            tau=0.95,
            n_envs=128,
            ppo_clip=0.2,
            train_range=('1990', '2016'),
            n_reuse_aux=0,
            n_reuse_policy=3,
            batch_size=16,
            lookahead=False,
        ),
        # , 'exponential'
        # gamma=tune.sample_from(lambda spec: None if spec.config.advantage_type == 'hyperbolic' \
        #     else np.random.choice([0.5, 0.8, 0.9, 0.95, 0.99])),

    )
    search_alg = None
    # current_best_params = [dict(
    #     entropy_weight=0.05,
    #     # weight_decay=1e-2,
    #     # lstm_size=1,
    #     # actor_size=1,
    #     # critic_size=1,
    # )]
    # search_alg = HyperOptSearch(metric='test_pnl', mode='max',
    #                             points_to_evaluate=current_best_params)

    analysis = tune.run(
        tune.with_parameters(run, data_dict=data_dict),
        # run,
        name='gamma_fixed_advantage',
        # local_dir='',
        num_samples=1000, fail_fast=True,
        resources_per_trial={"cpu": 1, "gpu": 0.124},
        search_alg=search_alg,
        config=config,
        queue_trials=True,
        # resume=True,
        # raise_on_failed_trial=False,
    )

    # print(analysis.results_df.sort_values('test_pnl'))
    # analysis_out = Path("~/rl_experiments/gym3_tests/tune_analysis").expanduser()
    # analysis_out.mkdir(exist_ok=True)
    # analysis.results_df.to_csv(str(analysis_out / "market_hyperopt.csv"))
    #
    # df = analysis.dataframe(metric='test_pnl', mode='max')
    # keys = [k for k in df.columns if 'config' in k]
    # from plotly import graph_objs as go
    #
    # for param in config.keys():
    #     fig = go.Figure()
    #     fig.add_scatter(x=df[param], y=df['test_pnl'], mode='markers', name='Test PnL')
    #     fig.add_scatter(x=df[param], y=df['train_pnl'], mode='markers', name='Train PnL')
    #     fig.write_html(str(analysis_out / f'{param.split("/")[-1]}_to_pnl.html'))
