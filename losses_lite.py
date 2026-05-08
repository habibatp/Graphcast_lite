# graphcast_lite/losses_lite.py
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch


LossAndDiagnostics = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


def normalized_latitude_weights(
    latitude: np.ndarray,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Même logique que GraphCast :
    poids latitude proportionnels à la surface réelle des cellules.
    """
    latitude = np.asarray(latitude, dtype=np.float32)

    if np.any(np.isclose(np.abs(latitude), 90.0)):
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = _weight_for_latitude_vector_without_poles(latitude)

    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=dtype, device=device)


def _weight_for_latitude_vector_without_poles(latitude: np.ndarray) -> np.ndarray:
    delta_latitude = abs(_check_uniform_spacing_and_get_delta(latitude))

    # Version Lite : on garde la formule cos(lat), même si ton domaine régional
    # ne couvre pas toute la Terre.
    return np.cos(np.deg2rad(latitude)).astype(np.float32)


def _weight_for_latitude_vector_with_poles(latitude: np.ndarray) -> np.ndarray:
    delta_latitude = abs(_check_uniform_spacing_and_get_delta(latitude))

    weights = (
        np.cos(np.deg2rad(latitude))
        * np.sin(np.deg2rad(delta_latitude / 2))
    )

    weights[0] = np.sin(np.deg2rad(delta_latitude / 4)) ** 2
    weights[-1] = np.sin(np.deg2rad(delta_latitude / 4)) ** 2

    return weights.astype(np.float32)


def _check_uniform_spacing_and_get_delta(vector: np.ndarray) -> float:
    diff = np.diff(vector)

    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError("Latitude vector is not uniformly spaced.")

    return float(diff[0])


def weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    latitude: Optional[np.ndarray] = None,
    channel_weights: Optional[torch.Tensor] = None,
) -> LossAndDiagnostics:
    """
    predictions : [B,C,H,W]
    targets     : [B,C,H,W]

    Retour :
        loss par batch : [B]
        diagnostics
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions={predictions.shape}, targets={targets.shape}"
        )

    loss = (predictions - targets) ** 2  # [B,C,H,W]

    if latitude is not None:
        lat_weights = normalized_latitude_weights(
            latitude,
            device=predictions.device,
            dtype=predictions.dtype,
        )  # [H]

        loss = loss * lat_weights.view(1, 1, -1, 1)

    if channel_weights is not None:
        channel_weights = channel_weights.to(
            device=predictions.device,
            dtype=predictions.dtype,
        )

        loss = loss * channel_weights.view(1, -1, 1, 1)

    loss_per_batch = loss.mean(dim=(1, 2, 3))  # [B]

    diagnostics = {
        "mse": loss_per_batch.detach(),
    }

    return loss_per_batch, diagnostics


def simple_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> LossAndDiagnostics:
    """
    MSE simple sans pondération.
    """
    loss = (predictions - targets) ** 2
    loss_per_batch = loss.mean(dim=(1, 2, 3))

    diagnostics = {
        "mse": loss_per_batch.detach(),
    }

    return loss_per_batch, diagnostics