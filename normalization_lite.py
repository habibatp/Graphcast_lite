# graphcast_lite/normalization_lite.py
import torch


def normalize(values, mean, std):
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (values - mean) / std


def unnormalize(values, mean, std):
    return values * std + mean


class InputsAndResiduals:
    """
    Wrapper de normalisation + residual learning.

    inputs:
        [B,T,C,H,W]

    targets:
        [B,C,H,W]

    Le modèle prédit :
        residual_norm = (target - last_input) / residual_std
    """

    def __init__(
        self,
        predictor,
        mean,
        std,
        residual_std,
    ):
        self.predictor = predictor
        self.mean = mean
        self.std = std
        self.residual_std = residual_std

    def _to_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.residual_std = self.residual_std.to(device)

    def normalize_inputs(self, inputs):
        return normalize(inputs, self.mean, self.std)

    def normalize_forcings(self, forcings):
        return forcings

    def target_to_normalized_residual(self, inputs, targets):
        last_input = inputs[:, -1]
        residual = targets - last_input
        return residual / self.residual_std

    def prediction_to_real_values(self, inputs, norm_prediction):
        last_input = inputs[:, -1]
        residual = norm_prediction * self.residual_std
        return last_input + residual

    def __call__(self, inputs, forcings=None):
        self._to_device(inputs.device)

        norm_inputs = self.normalize_inputs(inputs)

        norm_forcings = None
        if forcings is not None:
            norm_forcings = self.normalize_forcings(forcings)

        norm_prediction = self.predictor(norm_inputs, norm_forcings)

        prediction = self.prediction_to_real_values(
            inputs,
            norm_prediction,
        )

        return prediction

    def loss(self, inputs, targets, forcings=None, **kwargs):
        self._to_device(inputs.device)

        norm_inputs = self.normalize_inputs(inputs)

        norm_targets = self.target_to_normalized_residual(
            inputs,
            targets,
        )

        norm_forcings = None
        if forcings is not None:
            norm_forcings = self.normalize_forcings(forcings)

        return self.predictor.loss(
            norm_inputs,
            norm_targets,
            norm_forcings,
            **kwargs,
        )

    def loss_and_predictions(self, inputs, targets, forcings=None, **kwargs):
        self._to_device(inputs.device)

        norm_inputs = self.normalize_inputs(inputs)

        norm_targets = self.target_to_normalized_residual(
            inputs,
            targets,
        )

        norm_forcings = None
        if forcings is not None:
            norm_forcings = self.normalize_forcings(forcings)

        loss, norm_predictions = self.predictor.loss_and_predictions(
            norm_inputs,
            norm_targets,
            norm_forcings,
            **kwargs,
        )

        predictions = self.prediction_to_real_values(
            inputs,
            norm_predictions,
        )

        return loss, predictions