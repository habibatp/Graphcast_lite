# solar_radiation_lite.py
import dataclasses
from typing import Sequence, Optional, Union

import numpy as np
import pandas as pd

_DEFAULT_INTEGRATION_PERIOD = pd.Timedelta(hours=1)
_DEFAULT_NUM_INTEGRATION_BINS = 360

_JULIAN_YEAR_LENGTH_IN_DAYS = 365.25
_J2000_EPOCH = 2451545.0
_SECONDS_PER_DAY = 86400
_REFERENCE_TSI = 1361.0


@dataclasses.dataclass(frozen=True)
class OrbitalParameters:
    theta: np.ndarray
    rotational_phase: np.ndarray
    sin_declination: np.ndarray
    cos_declination: np.ndarray
    eq_of_time_seconds: np.ndarray
    solar_distance_au: np.ndarray


def era5_tsi_data():
    time = np.arange(1951.5, 2035.5, 1.0)
    tsi = 0.9965 * np.array([
        1365.7765, 1365.7676, 1365.6284, 1365.6564, 1365.7773,
        1366.3109, 1366.6681, 1366.6328, 1366.3828, 1366.2767,
        1365.9199, 1365.7484, 1365.6963, 1365.6976, 1365.7341,
        1365.9178, 1366.1143, 1366.1644, 1366.2476, 1366.2426,
        1365.9580, 1366.0525, 1365.7991, 1365.7271, 1365.5345,
        1365.6453, 1365.8331, 1366.2747, 1366.6348, 1366.6482,
        1366.6951, 1366.2859, 1366.1992, 1365.8103, 1365.6416,
        1365.6379, 1365.7899, 1366.0826, 1366.6479, 1366.5533,
        1366.4457, 1366.3021, 1366.0286, 1365.7971, 1365.6996,
        1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
        1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
        1365.8107, 1365.7240, 1365.6918,
        1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
        1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
        1365.8107, 1365.7240, 1365.6918,
        1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
        1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
        1365.8107, 1365.7240, 1365.6918,
    ], dtype=np.float32)
    return time, tsi


def get_tsi(timestamps: Sequence[Union[str, pd.Timestamp, np.datetime64]]) -> np.ndarray:
    tsi_time, tsi_values = era5_tsi_data()

    timestamps = pd.DatetimeIndex(timestamps)
    timestamps_date = pd.DatetimeIndex(timestamps.date)

    day_fraction = (timestamps - timestamps_date) / pd.Timedelta(days=1)
    year_length = 365 + timestamps.is_leap_year
    year_fraction = (timestamps.dayofyear - 1 + day_fraction) / year_length
    fractional_year = timestamps.year + year_fraction

    return np.interp(fractional_year, tsi_time, tsi_values).astype(np.float32)


def get_j2000_days(timestamp: pd.Timestamp) -> float:
    return timestamp.to_julian_date() - _J2000_EPOCH


def get_orbital_parameters(j2000_days: np.ndarray) -> OrbitalParameters:
    theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
    rotational_phase = np.mod(j2000_days, 1.0)

    rel = 1.7535 + 6.283076 * theta
    rem = 6.240041 + 6.283020 * theta
    rlls = 4.8951 + 6.283076 * theta

    one = np.ones_like(theta)

    sin_rel = np.sin(rel)
    cos_rel = np.cos(rel)
    sin_two_rel = np.sin(2.0 * rel)
    cos_two_rel = np.cos(2.0 * rel)
    sin_two_rlls = np.sin(2.0 * rlls)
    cos_two_rlls = np.cos(2.0 * rlls)
    sin_four_rlls = np.sin(4.0 * rlls)
    sin_rem = np.sin(rem)
    sin_two_rem = np.sin(2.0 * rem)

    rllls = np.dot(
        np.stack([one, theta, sin_rel, cos_rel, sin_two_rel, cos_two_rel], axis=-1),
        np.array([4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002]),
    )

    repsm = 0.409093

    sin_declination = np.sin(repsm) * np.sin(rllls)
    cos_declination = np.sqrt(1.0 - sin_declination ** 2)

    eq_of_time_seconds = np.dot(
        np.stack(
            [
                sin_two_rlls,
                sin_rem,
                sin_rem * cos_two_rlls,
                sin_four_rlls,
                sin_two_rem,
            ],
            axis=-1,
        ),
        np.array([591.8, -459.4, 39.5, -12.7, -4.8]),
    )

    solar_distance_au = np.dot(
        np.stack([one, sin_rel, cos_rel], axis=-1),
        np.array([1.0001, -0.0163, 0.0037]),
    )

    return OrbitalParameters(
        theta=theta,
        rotational_phase=rotational_phase,
        sin_declination=sin_declination,
        cos_declination=cos_declination,
        eq_of_time_seconds=eq_of_time_seconds,
        solar_distance_au=solar_distance_au,
    )


def get_solar_sin_altitude(
    op: OrbitalParameters,
    sin_latitude: np.ndarray,
    cos_latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    solar_time = op.rotational_phase + op.eq_of_time_seconds / _SECONDS_PER_DAY
    hour_angle = 2.0 * np.pi * solar_time + longitude

    return (
        cos_latitude * op.cos_declination * np.cos(hour_angle)
        + sin_latitude * op.sin_declination
    )


def get_radiation_flux(
    j2000_days: np.ndarray,
    sin_latitude: np.ndarray,
    cos_latitude: np.ndarray,
    longitude: np.ndarray,
    tsi: float,
) -> np.ndarray:
    op = get_orbital_parameters(j2000_days)

    solar_factor = (1.0 / op.solar_distance_au) ** 2

    sin_altitude = get_solar_sin_altitude(
        op=op,
        sin_latitude=sin_latitude,
        cos_latitude=cos_latitude,
        longitude=longitude,
    )

    return tsi * solar_factor * np.maximum(sin_altitude, 0.0)


def get_integrated_radiation(
    j2000_days: float,
    sin_latitude: np.ndarray,
    cos_latitude: np.ndarray,
    longitude: np.ndarray,
    tsi: float,
    integration_period: pd.Timedelta = _DEFAULT_INTEGRATION_PERIOD,
    num_integration_bins: int = _DEFAULT_NUM_INTEGRATION_BINS,
) -> np.ndarray:
    offsets = (
        pd.timedelta_range(
            start=-integration_period,
            end=pd.Timedelta(0),
            periods=num_integration_bins + 1,
        )
        / pd.Timedelta(days=1)
    ).to_numpy()

    fluxes = []

    for offset in offsets:
        flux = get_radiation_flux(
            j2000_days=np.array(j2000_days + offset, dtype=np.float32),
            sin_latitude=sin_latitude,
            cos_latitude=cos_latitude,
            longitude=longitude,
            tsi=tsi,
        )
        fluxes.append(flux)

    fluxes = np.stack(fluxes, axis=-1)

    dx = (integration_period / num_integration_bins) / pd.Timedelta(seconds=1)

    return np.trapz(fluxes, dx=dx, axis=-1).astype(np.float32)


def get_toa_incident_solar_radiation(
    timestamps: Sequence[Union[str, pd.Timestamp, np.datetime64]],
    latitude: np.ndarray,
    longitude: np.ndarray,
    integration_period: Union[str, pd.Timedelta, np.timedelta64] = _DEFAULT_INTEGRATION_PERIOD,
    num_integration_bins: int = _DEFAULT_NUM_INTEGRATION_BINS,
) -> np.ndarray:
    """
    Version Lite NumPy.

    Entrées :
        timestamps : liste de dates
        latitude   : [H]
        longitude  : [W]

    Sortie :
        radiation : [time, H, W]
    """

    integration_period = pd.Timedelta(integration_period)

    lat = np.radians(latitude).reshape(-1, 1).astype(np.float32)
    lon = np.radians(longitude).reshape(1, -1).astype(np.float32)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    tsi_values = get_tsi(timestamps)

    results = []

    for i, timestamp in enumerate(timestamps):
        timestamp = pd.Timestamp(timestamp)
        j2000_days = get_j2000_days(timestamp)

        radiation = get_integrated_radiation(
            j2000_days=j2000_days,
            sin_latitude=sin_lat,
            cos_latitude=cos_lat,
            longitude=lon,
            tsi=tsi_values[i],
            integration_period=integration_period,
            num_integration_bins=num_integration_bins,
        )

        results.append(radiation)

    return np.stack(results, axis=0).astype(np.float32)


if __name__ == "__main__":
    lat = np.linspace(46.0, 20.0, 64).astype(np.float32)
    lon = np.linspace(-15.0, 6.0, 64).astype(np.float32)

    timestamps = [
        pd.Timestamp("2020-01-01 00:00:00"),
        pd.Timestamp("2020-01-01 06:00:00"),
    ]

    tisr = get_toa_incident_solar_radiation(
        timestamps=timestamps,
        latitude=lat,
        longitude=lon,
        integration_period="1h",
        num_integration_bins=60,  # 60 pour test rapide, 360 pour plus proche ERA5
    )

    print("TISR shape:", tisr.shape)  # [time,H,W]
    print("min:", tisr.min(), "max:", tisr.max())