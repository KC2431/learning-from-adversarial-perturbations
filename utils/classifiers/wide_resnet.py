# rearranged from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn


class _WideBasic(nn.Module):
    def __init__(self, in_planes: int, planes: int, dropout_rate: float, stride: int = 1) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(
        self, 
        depth: int, 
        widen_factor: int, 
        dropout_rate: float, 
        n_class: int
    ) -> None:
        super().__init__()

        self.in_planes = 16

        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, padding=1)
        self.layer1 = self._wide_layer(_WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(_WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(_WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], n_class)

    def _wide_layer(
        self,
        block: Any, 
        planes: int, 
        num_blocks: int, 
        dropout_rate: float, 
        stride: int
    ) -> nn.Module:

        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.flatten(1)
        out = self.linear(out)
        return out