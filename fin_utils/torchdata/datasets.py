import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union

# 
# class FeatureLabelDataset(Dataset):
# 
#     def __init__(self, features: Union[pd.DataFrame, np.ndarray], labels: Union[pd.DataFrame, np.ndarray], name=None,
#                  index=None, include_index=True):
#         self.name = name
#         if index is not None:
#             self.index = index
#             assert index.shape[0] == features.shape[0], "index shape must be equal to features shape."
#             assert index.shape[0] == labels.shape[0], "index shape must be equal to labels shape."
#         else:
#             assert all(features.index == labels.index), "labels must have identical indexes to features.."
#             self.index = features.index.to_numpy().astype('long')
#         if isinstance(features, pd.core.generic.NDFrame):
#             features = features.values
#         if isinstance(labels, pd.core.generic.NDFrame):
#             labels = labels.values
#         self.features = torch.as_tensor(features)
#         self.labels = torch.as_tensor(labels)
#         self.include_index = include_index
# 
#     def __getitem__(self, item):
#         sample = {
#             'features': self.features[item],
#             'labels': self.labels[item],
#             'pair': self.name
#         }
#         if self.include_index:
#             sample['index'] = self.index[item]
#         return sample
# 
#     def __len__(self):
#         return self.features.shape[0]
# 
# 
# class WindowedFeatureLabelDataset(FeatureLabelDataset):
#     def __init__(self, *args, window_size, step_size=1, **kwargs):
#         super(WindowedFeatureLabelDataset, self).__init__(*args, **kwargs)
#         self.window_size = window_size
#         self.step_size = step_size
#         # self.index = self.index[window_size:]
# 
#     def __getitem__(self, item):
#         item *= self.step_size
#         sample = {
#             'features': self.features[item:item + self.window_size],
#             'labels': self.labels[item:item + self.window_size],
#             'pair': self.name
#         }
#         if self.include_index:
#             sample['index'] = self.index[item:item + self.window_size],
#             # sample['index'] = self.index[item],
# 
#         return sample
# 
#     def __len__(self):
#         return (self.features.shape[0] - self.window_size) // self.step_size
