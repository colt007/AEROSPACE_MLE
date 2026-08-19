import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from configs.config import BATCH_SIZE, RANDOM_SEED, TRAIN_SPLIT, WINDOW_SIZE
from src.data.datasetclass import CMAPSSDataset


def get_dataloaders(
    path: str,
    window_size: int = WINDOW_SIZE,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    random_seed: int = RANDOM_SEED):
    df = pd.read_parquet(path)

    unique_engine_ids = df["global_engine_id"].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_engine_ids)

    split_idx = int(len(unique_engine_ids) * train_split)
    train_engine_ids = unique_engine_ids[:split_idx]
    valid_engine_ids = unique_engine_ids[split_idx:]

    train_df = df[df["global_engine_id"].isin(train_engine_ids)].copy()
    valid_df = df[df["global_engine_id"].isin(valid_engine_ids)].copy()

    train_dataset = CMAPSSDataset(dataframe=train_df, window_size=window_size)
    valid_dataset = CMAPSSDataset(dataframe=valid_df, window_size=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Total engines      : {len(unique_engine_ids)}")
    print(f"Train engines      : {len(train_engine_ids)}")
    print(f"Validation engines : {len(valid_engine_ids)}")

    print(f"Train rows         : {len(train_df):,}")
    print(f"Validation rows    : {len(valid_df):,}")

    print(f"Train windows      : {len(train_dataset):,}")
    print(f"Validation windows : {len(valid_dataset):,}")

    return train_loader, valid_loader

    