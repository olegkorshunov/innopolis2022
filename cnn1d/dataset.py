from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


def get_df(path: str, train=True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    df = pd.read_csv(path)
    df = df.sort_index(axis=1)
    df.drop(['id', '.geo', 'area'], axis=1, inplace=True)
    if train:
        y = df[['crop']]
        df.drop(['crop'], axis=1, inplace=True)

    df.rename({c: pd.to_datetime(c.replace('nd_mean_', ''))
              for c in df.columns}, axis=1, inplace=True)
    df[df <= 0.0] = 0
    return (df.reset_index(drop=True).values, y.crop.reset_index(drop=True).values) if train else df.reset_index(drop=True).values


class CustomDataset(Dataset):

    def __init__(self, df: np.ndarray, y: np.ndarray = None, mode: str = 'Train', noise=False):
        """
        mode: Train/Val/Test
        """
        self._len = df.shape[0]

        df = np.copy(df)

        assert np.isnan(df).sum().sum() == 0, 'NAAAAAAAAAN'
        self.df = torch.unsqueeze(torch.tensor(df, dtype=torch.float32), dim=1)

        self.mode = mode
        if mode in {'Train', 'Val'}:
            self.y = torch.tensor(y, dtype=torch.long)
        self.noise = noise

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        data = {
            'X': self.df[idx],
        }
        if self.mode in {'Train', 'Val'}:
            data['y'] = self.y[idx]

        if self.noise:
            _noise = torch.normal(mean=0, std=0.006, size=self.df[idx].shape)
            data['X'] += _noise
            data['X'][data['X'] < 0] = 0
        return data
