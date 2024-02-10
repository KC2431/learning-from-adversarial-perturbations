import torch.nn.functional as F
from torch import Tensor, nn


class ConvNet(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.linear = nn.Linear(256, out_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x + F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x + F.relu(self.conv5(x))
        x = F.avg_pool2d(x, 9)
        x = x.flatten(1)
        return self.linear(x)