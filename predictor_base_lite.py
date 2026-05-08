# graphcast_lite/predictor_base_lite.py
import abc
from typing import Tuple, Dict, Any

import torch


LossAndDiagnostics = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


class Predictor(nn.Module, abc.ABC):
    """
    Interface de base pour les modèles GraphCast Lite.
    """

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        forcings: torch.Tensor | None = None,
        **optional_kwargs,
    ) -> torch.Tensor:
        """
        inputs   : [B,T,C,H,W]
        forcings : [B,F,H,W] ou None
        returns  : [B,C,H,W]
        """

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        forcings: torch.Tensor | None = None,
        **optional_kwargs,
    ) -> LossAndDiagnostics:
        batch_size = inputs.shape[0]
        dummy_loss = torch.zeros(batch_size, device=inputs.device)
        return dummy_loss, {}

    def loss_and_predictions(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        forcings: torch.Tensor | None = None,
        **optional_kwargs,
    ) -> Tuple[LossAndDiagnostics, torch.Tensor]:
        raise NotImplementedError