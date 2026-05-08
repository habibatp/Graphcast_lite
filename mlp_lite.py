# graphcast_lite/mlp_lite.py
import torch
import torch.nn as nn


class LinearNormConditioning(nn.Module):
    """
    Version PyTorch de LinearNormConditioning.

    Formule :
        output = inputs * scale + offset

    où scale et offset sont calculés à partir de norm_conditioning.
    """

    def __init__(self, input_size: int, conditioning_size: int):
        super().__init__()

        self.linear = nn.Linear(conditioning_size, 2 * input_size)

        nn.init.trunc_normal_(self.linear.weight, std=1e-8)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor, norm_conditioning: torch.Tensor):
        feature_size = inputs.shape[-1]

        scale_offset = self.linear(norm_conditioning)

        scale_minus_one, offset = torch.split(
            scale_offset,
            feature_size,
            dim=-1,
        )

        scale = scale_minus_one + 1.0

        return inputs * scale + offset