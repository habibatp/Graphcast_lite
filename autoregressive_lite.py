# graphcast_lite/autoregressive_lite.py
import torch


class AutoregressivePredictor:
    """
    Wrapper autoregressive pour GraphCast Lite.

    Il transforme un modèle one-step :
        t -> t+1

    en modèle multi-step :
        t -> t+1 -> t+2 -> ...
    """

    def __init__(
        self,
        predictor,
        noise_level=None,
        gradient_checkpointing=False,
    ):
        self.predictor = predictor
        self.noise_level = noise_level
        self.gradient_checkpointing = gradient_checkpointing

    def _add_noise(self, inputs):
        if self.noise_level is None:
            return inputs

        noise = torch.randn_like(inputs) * self.noise_level
        return inputs + noise

    def _update_inputs(self, inputs, next_prediction):
        """
        inputs:
            [B,T,C,H,W]

        next_prediction:
            [B,C,H,W]

        return:
            [B,T,C,H,W]
        """
        next_prediction = next_prediction.unsqueeze(1)

        return torch.cat(
            [inputs[:, 1:], next_prediction],
            dim=1,
        )

    def __call__(self, inputs, forcings_sequence=None, horizon=1):
        """
        inputs:
            [B,T,C,H,W]

        forcings_sequence:
            None
            ou [B,horizon,F,H,W]

        return:
            [B,horizon,C,H,W]
        """
        predictions = []
        current_inputs = inputs

        for step in range(horizon):
            if forcings_sequence is not None:
                forcing_step = forcings_sequence[:, step]
            else:
                forcing_step = None

            pred = self.predictor(
                current_inputs,
                forcing_step,
            )

            predictions.append(pred)

            current_inputs = self._update_inputs(
                current_inputs,
                pred,
            )

        return torch.stack(predictions, dim=1)

    def loss(self, inputs, targets_sequence, forcings_sequence=None):
        """
        targets_sequence:
            [B,horizon,C,H,W]
        """
        horizon = targets_sequence.shape[1]

        current_inputs = self._add_noise(inputs)

        losses = []
        diagnostics_all = {}

        for step in range(horizon):
            target_step = targets_sequence[:, step]

            if forcings_sequence is not None:
                forcing_step = forcings_sequence[:, step]
            else:
                forcing_step = None

            loss_step, diagnostics = self.predictor.loss(
                current_inputs,
                target_step,
                forcing_step,
            )

            losses.append(loss_step)

            for k, v in diagnostics.items():
                diagnostics_all.setdefault(k, []).append(v)

            with torch.no_grad():
                pred_step = self.predictor(
                    current_inputs,
                    forcing_step,
                )

            current_inputs = self._update_inputs(
                current_inputs,
                pred_step,
            )

        loss = torch.stack(losses, dim=0).mean(dim=0)

        diagnostics_final = {
            k: torch.stack(v, dim=0).mean(dim=0)
            for k, v in diagnostics_all.items()
        }

        return loss, diagnostics_final

    def loss_and_predictions(self, inputs, targets_sequence, forcings_sequence=None):
        horizon = targets_sequence.shape[1]

        current_inputs = self._add_noise(inputs)

        losses = []
        predictions = []
        diagnostics_all = {}

        for step in range(horizon):
            target_step = targets_sequence[:, step]

            if forcings_sequence is not None:
                forcing_step = forcings_sequence[:, step]
            else:
                forcing_step = None

            (loss_step, diagnostics), pred_step = self.predictor.loss_and_predictions(
                current_inputs,
                target_step,
                forcing_step,
            )

            losses.append(loss_step)
            predictions.append(pred_step)

            for k, v in diagnostics.items():
                diagnostics_all.setdefault(k, []).append(v)

            current_inputs = self._update_inputs(
                current_inputs,
                pred_step,
            )

        loss = torch.stack(losses, dim=0).mean(dim=0)

        diagnostics_final = {
            k: torch.stack(v, dim=0).mean(dim=0)
            for k, v in diagnostics_all.items()
        }

        predictions = torch.stack(predictions, dim=1)

        return (loss, diagnostics_final), predictions