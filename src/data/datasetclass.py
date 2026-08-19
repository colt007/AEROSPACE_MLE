import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from configs.config import WINDOW_SIZE


class ToTensor:
    def __call__(self, features, target):
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return features, target


class CMAPSSDataset(Dataset):
    def __init__(self, parquet_path=None, dataframe=None, window_size=WINDOW_SIZE, transform=None):
        if dataframe is not None:
            df = dataframe
        elif parquet_path is not None:
            df = pd.read_parquet(parquet_path)
        else:
            raise ValueError("You must provide either a parquet_path or a dataframe.")

        self.window_size = window_size
        self.transform = transform
        self.global_features = []
        self.global_targets = []

        feature_columns = [
            col for col in df.columns
            if col not in ["time_cycle", "RUL", "global_engine_id"]
        ]

        for _, engine_df in df.groupby("global_engine_id"):
            engine_features = engine_df[feature_columns].astype(np.float32).values
            engine_targets = engine_df["RUL"].astype(np.float32).values

            num_cycles = len(engine_df)
            num_windows = num_cycles - window_size + 1

            if num_windows <= 0:
                continue

            for i in range(num_windows):
                self.global_features.append(engine_features[i:i + window_size])
                self.global_targets.append(engine_targets[i + window_size - 1])

        self.global_features = np.asarray(self.global_features, dtype=np.float32)
        self.global_targets = np.asarray(self.global_targets, dtype=np.float32)

    def __len__(self):
        return len(self.global_features)

    def __getitem__(self, idx):
        features = self.global_features[idx]
        target = self.global_targets[idx]

        if self.transform is not None:
            features, target = self.transform(features, target)

        return features, target
