# data_utils_lite.py
import os
import re
import glob
from typing import Any, Sequence, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from solar_radiation_lite import get_toa_incident_solar_radiation

TimedeltaLike = Any
TargetLeadTimes = Union[TimedeltaLike, Sequence[TimedeltaLike], slice]

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"
TISR = "toa_incident_solar_radiation"

_DERIVED_VARS = {
    DAY_PROGRESS,
    f"{DAY_PROGRESS}_sin",
    f"{DAY_PROGRESS}_cos",
    YEAR_PROGRESS,
    f"{YEAR_PROGRESS}_sin",
    f"{YEAR_PROGRESS}_cos",
}


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    years_since_epoch = (
        seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
    )
    return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(seconds_since_epoch: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)

    day_progress = np.mod(
        day_progress_greenwich[..., np.newaxis] + longitude_offsets,
        1.0
    )
    return day_progress.astype(np.float32)


def featurize_progress(progress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    progress_phase = progress * (2 * np.pi)
    return (
        progress.astype(np.float32),
        np.sin(progress_phase).astype(np.float32),
        np.cos(progress_phase).astype(np.float32),
    )


def get_seconds_since_epoch(datetimes: Sequence[pd.Timestamp]) -> np.ndarray:
    datetimes = pd.DatetimeIndex(datetimes)
    return datetimes.astype("datetime64[s]").astype(np.int64).to_numpy()


def parse_datetime_from_filename(path: str) -> pd.Timestamp:
    """
    Pour fichiers du type :
        1986_01_000.npy

    On suppose que 000 = premier pas 6h du mois.
    Donc :
        000 -> jour 1 à 00h
        001 -> jour 1 à 06h
        002 -> jour 1 à 12h
        003 -> jour 1 à 18h
        004 -> jour 2 à 00h
    """
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)

    if len(nums) < 3:
        raise ValueError(f"Nom de fichier invalide : {name}")

    year = int(nums[0])
    month = int(nums[1])
    step = int(nums[2])

    base = pd.Timestamp(year=year, month=month, day=1, hour=0)
    return base + pd.Timedelta(hours=6 * step)


def sort_npy_files(data_dir: str) -> list[str]:
    files = glob.glob(os.path.join(data_dir, "*.npy"))

    def key(path):
        name = os.path.basename(path)
        nums = re.findall(r"\d+", name)
        return tuple(map(int, nums))

    return sorted(files, key=key)


def build_forcing_channels(
    target_datetime: pd.Timestamp,
    lat: np.ndarray,
    lon: np.ndarray,
    include_progress: bool = True,
    include_tisr_proxy: bool = True,
) -> np.ndarray:
    """
    Construit les forcings pour un seul target time.

    Sortie :
        forcings [F, H, W]

    F par défaut :
        year_progress
        year_progress_sin
        year_progress_cos
        day_progress
        day_progress_sin
        day_progress_cos
        tisr_proxy
    """
    H = len(lat)
    W = len(lon)

    seconds = get_seconds_since_epoch([target_datetime])  # [1]

    channels = []

    if include_progress:
        year_progress = get_year_progress(seconds)  # [1]
        yp, yp_sin, yp_cos = featurize_progress(year_progress)

        day_progress_lon = get_day_progress(seconds, lon)  # [1, W]
        dp, dp_sin, dp_cos = featurize_progress(day_progress_lon)

        year_map = np.ones((H, W), dtype=np.float32) * yp[0]
        year_sin_map = np.ones((H, W), dtype=np.float32) * yp_sin[0]
        year_cos_map = np.ones((H, W), dtype=np.float32) * yp_cos[0]

        day_map = np.tile(dp[0][None, :], (H, 1)).astype(np.float32)
        day_sin_map = np.tile(dp_sin[0][None, :], (H, 1)).astype(np.float32)
        day_cos_map = np.tile(dp_cos[0][None, :], (H, 1)).astype(np.float32)

        channels.extend([
            year_map,
            year_sin_map,
            year_cos_map,
            day_map,
            day_sin_map,
            day_cos_map,
        ])

    if include_tisr_proxy:
        tisr = get_toa_incident_solar_radiation(
            timestamps=[target_datetime],
            latitude=lat,
            longitude=lon,
            integration_period="1h",
            num_integration_bins=60,   # 60 rapide, 360 plus précis
        )[0]
        channels.append(tisr.astype(np.float32))

    return np.stack(channels, axis=0).astype(np.float32)


def process_target_lead_times_and_get_steps(
    target_lead_times: TargetLeadTimes,
    data_time_step_hours: int = 6,
) -> Tuple[list[int], int]:
    """
    Version npy de _process_target_lead_times_and_get_duration.

    Convertit les lead times en indices de fichiers.

    Exemple :
        target_lead_times="6h"  -> [1]
        target_lead_times="12h" -> [2]
    """
    if isinstance(target_lead_times, slice):
        start = pd.Timedelta(target_lead_times.start or "6h")
        stop = pd.Timedelta(target_lead_times.stop)
        step = pd.Timedelta(target_lead_times.step or f"{data_time_step_hours}h")

        lead_times = []
        current = start
        while current <= stop:
            lead_times.append(current)
            current += step
    else:
        if not isinstance(target_lead_times, (list, tuple, set)):
            target_lead_times = [target_lead_times]
        lead_times = [pd.Timedelta(x) for x in target_lead_times]
        lead_times.sort()

    step_hours = pd.Timedelta(hours=data_time_step_hours)

    lead_steps = []
    for lt in lead_times:
        if lt % step_hours != pd.Timedelta(0):
            raise ValueError(f"Lead time {lt} n'est pas multiple de {step_hours}")
        lead_steps.append(int(lt / step_hours))

    return lead_steps, max(lead_steps)


def extract_input_target_times_from_files(
    files: list[str],
    idx: int,
    input_steps: int,
    target_lead_times: TargetLeadTimes = "6h",
    data_time_step_hours: int = 6,
) -> Tuple[list[str], list[str]]:
    """
    Equivalent Lite de extract_input_target_times.

    Retourne :
        input_files  : fichiers historiques
        target_files : fichiers futurs demandés
    """
    lead_steps, max_lead_step = process_target_lead_times_and_get_steps(
        target_lead_times,
        data_time_step_hours=data_time_step_hours,
    )

    input_start = idx
    input_end = idx + input_steps

    input_files = files[input_start:input_end]

    reference_index = input_end - 1
    target_files = [files[reference_index + step] for step in lead_steps]

    return input_files, target_files


def extract_inputs_targets_forcings_npy(
    files: list[str],
    idx: int,
    lat: np.ndarray,
    lon: np.ndarray,
    input_steps: int = 2,
    target_lead_times: TargetLeadTimes = "6h",
    data_time_step_hours: int = 6,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    residual_std: Optional[np.ndarray] = None,
    normalize: bool = True,
    use_residual: bool = True,
    include_forcings: bool = True,
    mmap: bool = True,
):
    """
    Equivalent Lite de extract_inputs_targets_forcings.

    Retourne :
        inputs   : torch.Tensor [T, C, H, W]
        targets  : torch.Tensor [C, H, W] si un seul lead time
        forcings : torch.Tensor [F, H, W] si un seul lead time
    """
    input_files, target_files = extract_input_target_times_from_files(
        files=files,
        idx=idx,
        input_steps=input_steps,
        target_lead_times=target_lead_times,
        data_time_step_hours=data_time_step_hours,
    )

    def load_frame(path):
        arr = np.load(path, mmap_mode="r" if mmap else None)
        return np.asarray(arr, dtype=np.float32)

    inputs = np.stack([load_frame(f) for f in input_files], axis=0)  # [T,C,H,W]
    targets = np.stack([load_frame(f) for f in target_files], axis=0)  # [L,C,H,W]

    if normalize:
        if mean is None or std is None:
            raise ValueError("mean/std nécessaires si normalize=True")

        std = np.where(std == 0, 1.0, std)
        inputs_norm = (inputs - mean) / std

        if use_residual:
            if residual_std is None:
                raise ValueError("residual_std nécessaire si use_residual=True")
            residual_std = np.where(residual_std == 0, 1.0, residual_std)

            last_input = inputs[-1]  # [C,H,W]
            targets_out = (targets - last_input[None, ...]) / residual_std
        else:
            targets_out = (targets - mean) / std

        inputs = inputs_norm
        targets = targets_out

    forcings = None
    if include_forcings:
        forcing_list = []
        for target_file in target_files:
            dt = parse_datetime_from_filename(target_file)
            forcing = build_forcing_channels(
                target_datetime=dt,
                lat=lat,
                lon=lon,
                include_progress=True,
                include_tisr_proxy=True,
            )
            forcing_list.append(forcing)

        forcings = np.stack(forcing_list, axis=0)  # [L,F,H,W]

    # Si one-step, enlever dimension L pour rester simple.
    if targets.shape[0] == 1:
        targets = targets[0]  # [C,H,W]
        if forcings is not None:
            forcings = forcings[0]  # [F,H,W]

    inputs = torch.from_numpy(inputs.copy()).float()
    targets = torch.from_numpy(targets.copy()).float()

    if forcings is not None:
        forcings = torch.from_numpy(forcings.copy()).float()

    return inputs, targets, forcings


class ERA5GraphCastLiteDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch final.

    Chaque sample retourne :
        X : [T,C,H,W]
        Y : [C,H,W]
        F : [F,H,W]
    """

    def __init__(
        self,
        data_dir: str,
        lat: np.ndarray,
        lon: np.ndarray,
        input_steps: int = 2,
        target_lead_times: TargetLeadTimes = "6h",
        data_time_step_hours: int = 6,
        stats_dir: Optional[str] = None,
        normalize: bool = True,
        use_residual: bool = True,
        include_forcings: bool = True,
        mmap: bool = True,
    ):
        self.files = sort_npy_files(data_dir)
        self.lat = lat.astype(np.float32)
        self.lon = lon.astype(np.float32)
        self.input_steps = input_steps
        self.target_lead_times = target_lead_times
        self.data_time_step_hours = data_time_step_hours
        self.normalize = normalize
        self.use_residual = use_residual
        self.include_forcings = include_forcings
        self.mmap = mmap

        _, max_lead_step = process_target_lead_times_and_get_steps(
            target_lead_times,
            data_time_step_hours=data_time_step_hours,
        )

        self.length = len(self.files) - input_steps - max_lead_step + 1
        if self.length <= 0:
            raise ValueError("Pas assez de fichiers .npy pour créer le dataset.")

        self.mean = None
        self.std = None
        self.residual_std = None

        if normalize:
            if stats_dir is None:
                raise ValueError("stats_dir nécessaire si normalize=True")

            self.mean = np.load(os.path.join(stats_dir, "mean.npy")).astype(np.float32)
            self.std = np.load(os.path.join(stats_dir, "std.npy")).astype(np.float32)

            if use_residual:
                self.residual_std = np.load(
                    os.path.join(stats_dir, "residual_std.npy")
                ).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inputs, targets, forcings = extract_inputs_targets_forcings_npy(
            files=self.files,
            idx=idx,
            lat=self.lat,
            lon=self.lon,
            input_steps=self.input_steps,
            target_lead_times=self.target_lead_times,
            data_time_step_hours=self.data_time_step_hours,
            mean=self.mean,
            std=self.std,
            residual_std=self.residual_std,
            normalize=self.normalize,
            use_residual=self.use_residual,
            include_forcings=self.include_forcings,
            mmap=self.mmap,
        )

        if self.include_forcings:
            return inputs, forcings, targets

        return inputs, targets


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\user\Desktop\Graphcast_Project\ERA5_np_float32_1986_2026"
    STATS_DIR = r"C:\Users\user\Desktop\Graphcast_Project\ERA5_np_stats"

    lat = np.linspace(46.0, 20.0, 64).astype(np.float32)
    lon = np.linspace(-15.0, 6.0, 64).astype(np.float32)

    dataset = ERA5GraphCastLiteDataset(
        data_dir=DATA_DIR,
        stats_dir=STATS_DIR,
        lat=lat,
        lon=lon,
        input_steps=2,
        target_lead_times="6h",
        include_forcings=True,
        normalize=True,
        use_residual=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    x, f, y = next(iter(loader))

    print("X:", x.shape)  # [B,T,C,H,W]
    print("F:", f.shape)  # [B,F,H,W]
    print("Y:", y.shape)  # [B,C,H,W]