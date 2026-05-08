# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_utils_lite import ERA5GraphCastLiteDataset


def create_dataloaders(
    data_dir,
    stats_dir,
    input_steps=2,
    target_lead_times="6h",
    batch_size=1,
    num_workers=0,
    include_forcings=True,
):
    lat = np.linspace(46.0, 20.0, 64).astype(np.float32)
    lon = np.linspace(-15.0, 6.0, 64).astype(np.float32)

    dataset = ERA5GraphCastLiteDataset(
        data_dir=data_dir,
        stats_dir=stats_dir,
        lat=lat,
        lon=lon,
        input_steps=input_steps,
        target_lead_times=target_lead_times,
        include_forcings=include_forcings,
        normalize=True,
        use_residual=True,
        mmap=True,
    )

    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\user\Desktop\Graphcast_Project\ERA5_np_float32_1986_2026"
    STATS_DIR = r"C:\Users\user\Desktop\Graphcast_Project\ERA5_np_stats"

    train_loader, val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        stats_dir=STATS_DIR,
        input_steps=2,
        target_lead_times="6h",
        batch_size=1,
        num_workers=0,
        include_forcings=True,
    )

    x, f, y = next(iter(train_loader))

    print("✅ Test DataLoader OK")
    print("X shape:", x.shape)  # [B,T,C,H,W]
    print("F shape:", f.shape)  # [B,F,H,W]
    print("Y shape:", y.shape)  # [B,C,H,W]