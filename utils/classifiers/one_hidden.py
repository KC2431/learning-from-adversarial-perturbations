import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class OneHiddenNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        assert hidden_dim % 2 == 0
        super().__init__()

        self.linear = nn.Linear(in_dim, hidden_dim, False)

        const = torch.empty(1, hidden_dim)

        half_hidden_dim = hidden_dim // 2
        const[:, :half_hidden_dim] = 1 / math.sqrt(hidden_dim)
        const[:, half_hidden_dim:] = - 1 / math.sqrt(hidden_dim)

        self.register_buffer('const', const)
        self.const: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.leaky_relu(self.linear(x)), self.const).view(-1)