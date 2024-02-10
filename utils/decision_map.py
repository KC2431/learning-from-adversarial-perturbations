import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.classifiers import OneHiddenNet


def get_axis_vec(classifier_weight: Tensor) -> Tuple[Tensor, Tensor]:
    half_hidden_dim = len(classifier_weight) // 2
    axis_vec_1 = F.normalize(classifier_weight[:half_hidden_dim].mean(0), dim=0) # (in_dim,)
    axis_vec_2 = F.normalize(classifier_weight[half_hidden_dim:].mean(0), dim=0)
    return axis_vec_1, axis_vec_2


def get_inputs_for_decision_map(
    axis_vec_1: Tensor, 
    axis_vec_2: Tensor, 
    resolution: int, 
    limit: float,
) -> Tensor:
    
    s = torch.linspace(-limit, limit, resolution)
    # (resolution, resolution), (resolution, resolution)
    x, y = torch.meshgrid(s, s, indexing='xy')
    y = y.flip(0)

    in_dim = len(axis_vec_1)
    expanded_axis_vec_1 = axis_vec_1.expand((resolution, resolution, in_dim))
    expanded_axis_vec_2 = axis_vec_2.expand((resolution, resolution, in_dim))

    inputs = x[:, :, None] * expanded_axis_vec_1 + y[:, :, None] * expanded_axis_vec_2
    return inputs.view(resolution**2, in_dim)


def get_decision_map(classifier: OneHiddenNet, inputs: Tensor) -> Tensor:
    resolution = int(math.sqrt(len(inputs)))
    return classifier(inputs).sign().view(resolution, resolution).cpu()