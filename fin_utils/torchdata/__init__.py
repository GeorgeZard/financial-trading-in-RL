from fin_utils.torchdata.datasets import FeatureLabelDataset, WindowedFeatureLabelDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path

from fin_utils.read_feather import load_feather_dir
from fin_utils.candles.features import construct_features
from fin_utils.candles.labels import construct_labels

from typing import List, Union, Dict
import pandas as pd
import torch


def construct_windowed_dataloader(
        feather_folder: Union[str, Path], feature_config: List[Dict],
        label_config: Dict, pairs=None, window_size: int = 32,
        candle_length: Union[str, pd.Timedelta] = '4H',
        train_end: Union[str, pd.Timestamp] = '2016',

):
    candle_dict = load_feather_dir(feather_folder, pairs=pairs, resample=candle_length)
    feature_dict, ret_stats = construct_features(
        candle_dict, train_end, normlize_with_mean=True,
        feature_config=feature_config)
    label_dict = construct_labels(candle_dict, label_config)

    datasets = []

    for k in feature_dict.keys():
        datasets.append(WindowedFeatureLabelDataset(
            window_size=window_size,
            features=feature_dict[k], name=k,
            labels=label_dict[k],
        ))
    concat_dataset = ConcatDataset(datasets)
    # sampler = SequentialDateRangeSampler(concat_dataset,
    #                                      '2000', '2016', batch_size=16)
    # custom_dataloader = DataLoader(concat_dataset,
    #                                batch_sampler=sampler)
    # for b in tqdm(custom_dataloader):
    #     pass
    return concat_dataset


drs_compatible_dataset_types = (FeatureLabelDataset, WindowedFeatureLabelDataset)


class DateRangeSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start: Union[str, pd.Timestamp],
                 stop: Union[str, pd.Timestamp], batch_size: int):
        super(DateRangeSampler, self).__init__(data_source)
        self.start = pd.to_datetime(start).to_numpy()
        self.stop = pd.to_datetime(stop).to_numpy()
        self.data_source = data_source
        self.batch_size = batch_size
        self.ranges = []
        if isinstance(data_source, ConcatDataset):
            assert all([isinstance(d, drs_compatible_dataset_types) for d in data_source.datasets])
            for ds, cs in zip(data_source.datasets, data_source.cumulative_sizes):
                start_idx, stop_idxs = ds.index.searchsorted((
                    self.start.astype('long'), self.stop.astype('long')
                )) + (cs - len(ds))
                self.ranges.append((int(start_idx), int(stop_idxs)))
        else:
            assert isinstance(data_source, drs_compatible_dataset_types)
            start_idx, stop_idxs = data_source.index.searchsorted((start, stop))
            self.ranges.append((int(start_idx), int(stop_idxs)))

    def __len__(self):
        return sum([r[1] - r[0] for r in self.ranges])


class RandomDateRangeSampler(DateRangeSampler):
    def __iter__(self):
        perms = []
        for i, r in enumerate(self.ranges):
            perms.append(torch.randperm(r[1] - r[0]) + r[0])
        perms = torch.cat(perms, dim=0)
        for i in range(self.batch_size, perms.shape[0], self.batch_size):
            yield perms[i - self.batch_size:i]


class SequentialDateRangeSampler(DateRangeSampler):
    def __iter__(self):
        cur_idxs = []
        # TODO: MAKE THIS FASTER
        for r in self.ranges:
            for i in range(r[0], r[1]):
                cur_idxs.append(i)
                if len(cur_idxs) == self.batch_size:
                    yield torch.as_tensor(cur_idxs)
                    del cur_idxs[:]


class WindowFeatureDataModule(LightningDataModule):
    def __init__(self, data_dir, data_config: Dict, train_batch_size=16):
        super(WindowFeatureDataModule, self).__init__()
        self.data_dir = data_dir
        self.data_config = data_config.copy()
        self.train_range = self.data_config.pop('train_range')
        self.test_range = self.data_config.pop('test_range')
        self.valid_range = self.data_config.pop('valid_range', None)
        self.train_batch_size = train_batch_size

    def setup(self, stage=None):
        self.dataset = construct_windowed_dataloader(
            feather_folder=self.data_dir, **self.data_config
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_sampler=RandomDateRangeSampler(self.dataset, self.train_range[0], self.train_range[1],
                                                 batch_size=self.train_batch_size))

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset,
            batch_sampler=SequentialDateRangeSampler(self.dataset, self.test_range[0], self.test_range[1],
                                                     batch_size=self.train_batch_size))

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.valid_range:
            return DataLoader(
                self.dataset,
                batch_sampler=SequentialDateRangeSampler(self.dataset, self.valid_range[0], self.valid_range[1],
                                                         batch_size=self.train_batch_size))
        else:
            return self.test_dataloader(*args, **kwargs)


def build_independant_histograms_from_candles(
        candles_dict: Dict[str, pd.DataFrame],
        window: Union[str, pd.Timedelta],
        step: Union[str, pd.Timedelta],
        bin_range=0.05, nbins=100):
    from fin_histogram import build_independant_histograms
    hist_dict = dict()
    for k, candles in candles_dict.items():
        hist_dict[k] = build_independant_histograms(candles, window=window, step=step, bin_range=bin_range)
    return hist_dict


LOADER_FUNCS = dict(
    load_feather_dir=load_feather_dir,
)

FEATURE_EXTRACTORS = dict(
    candle_feature_constructor=construct_features,
    candle_histogram_constructor=build_independant_histograms_from_candles
)
LABEL_EXTRACTORS = dict(
    candle_label_constructor=construct_labels
)


def data_builder(pipelines_config):
    features = []
    labels = []
    for data_conf_group in pipelines_config:
        loader_func = LOADER_FUNCS[data_conf_group['data_loader']]
        loader_kwargs = data_conf_group.get('data_loader_conf', {})
        cur_data = loader_func(**loader_kwargs)
        if 'feature_extractor' in data_conf_group:
            feature_extractor = FEATURE_EXTRACTORS[data_conf_group['feature_extractor']]
            feature_extractor_kwargs = data_conf_group.get('feature_config', {})
            features.append(feature_extractor(cur_data, **feature_extractor_kwargs))
        if 'label_extractor' in data_conf_group:
            label_extractor = LABEL_EXTRACTORS[data_conf_group['label_extractor']]
            label_extractor_kwargs = data_conf_group.get('label_config', {})
            labels.append(label_extractor(cur_data, **label_extractor_kwargs))
    concated_features = dict()
    concated_labels = dict()
    for pair in features[0].keys():
        pair_features = pd.concat([features[i][pair] for i in range(len(features))], join='inner', axis=1)
        pair_labels = pd.concat([labels[i][pair] for i in range(len(labels))], join='inner', axis=1)
        pair_features, pair_labels = pair_features.align(pair_labels, axis=0, join='inner')
        concated_features[pair], concated_labels[pair] = pair_features, pair_labels
    return concated_features, concated_labels


def create_datasets_from_dicts(feature_dict, label_dict, ds_range, window_size=50):
    datasets = []

    for k in feature_dict.keys():
        idx_slice = slice(feature_dict[k].index.searchsorted(ds_range[0]),
                          feature_dict[k].index.searchsorted(ds_range[1]))
        datasets.append(WindowedFeatureLabelDataset(
            window_size=window_size,
            features=feature_dict[k][idx_slice], name=k,
            labels=label_dict[k][idx_slice], include_index=True
        ))
    concat_dataset = ConcatDataset(datasets)
    return concat_dataset


if __name__ == '__main__':
    feather_folder = Path('~/financial_data/histdata_feathers')
    pairs = ['eurusd', 'gbpjpy']
    window = '1H'
    candle_feature_config = [
        dict(name='int_bar_changes', func_name='inter_bar_changes',
             columns=['close', 'high', 'low'],
             use_pct=True),
        dict(name='int_bar_changes_10', func_name='inter_bar_changes',
             columns=['close', 'high', 'low'], use_pct=True,
             smoothing_window=10),
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
    ]
    label_config = dict(name='triple_barrier_labels',
                        parameters=dict(
                            target_profit=0.01,
                            barrier_length=10,
                            use_hl=False,
                        ))
    new_config = [dict(
        # Config on how to load the inital data
        type='candle_features',
        data_loader='load_feather_dir',  # which function to run to load the data
        data_loader_conf=dict(  # the parameters of the data loading function.
            path='~/financial_data/histdata_feathers',
            pairs=pairs, resample=window
        ),
        feature_extractor="candle_feature_constructor",  # function that receives the loaded data to create the features
        feature_config=dict(feature_config=candle_feature_config, train_end='2016',
                            return_stats=False),  # parameters of function
        label_extractor="candle_label_constructor",
        label_config=dict(label_config=label_config),
    ), dict(
        type='histograms', data_loader='load_feather_dir',
        data_loader_conf=dict(  # the parameters of the data loading function.
            path='~/financial_data/histdata_feathers',
            pairs=pairs, resample='1min'
        ),
        feature_extractor="candle_histogram_constructor",
        feature_config=dict(
            window="12H", step=window,
            bin_range=0.05, nbins=100)
    )]
    features, labels = data_builder(new_config)
    create_datasets_from_dicts(features, labels)

