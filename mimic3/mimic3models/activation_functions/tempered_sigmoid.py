from torch import Tensor, sigmoid
from torch.nn import Module

"""
Code modified from the jax implementation in
Tempered Sigmoid Activations for
Deep Learning with Differential Privacy (2021)
by Papernot et. al
"""


def tempered_sigmoid(x, scale=2.0, inverse_temp=2.0, offset=1.0):
    return scale * sigmoid(inverse_temp * x) - offset


class Temperedsigmoid(Module):
    def __init__(
        self, scale: float = 2, inverse_temperature: float = 2, offset: float = 1
    ):
        super(Temperedsigmoid, self).__init__()
        self.scale = scale
        self.inverse_temp = inverse_temperature
        self.offset = offset

    def forward(self, input: Tensor) -> Tensor:
        return tempered_sigmoid(
            input, scale=self.scale, inverse_temp=self.inverse_temp, offset=self.offset
        )
