import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from typing import Any
import torch.nn.functional as F

class BinaryResNet(nn.Module):
    def __init__(self, resnet_type):
        super(BinaryResNet, self).__init__()

        assert resnet_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

        self.models_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        self.resnet = self.models_dict[resnet_type](weights=None)  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Change FC layer to 1 output

    def forward(self, x):
        x = self.resnet(x)
        return x.view(-1)
    
class BinaryTrainedResNet(nn.Module):
    def __init__(self, resnet_type):
        super(BinaryTrainedResNet, self).__init__()

        assert resnet_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

        self.models_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        self.resnet = self.models_dict[resnet_type](weights='DEFAULT')  # pre-trained
        #for parameter in self.resnet.parameters():
        #    parameter.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # Change FC layer to 1 output

    def forward(self, x):
        x = self.resnet(x)
        return x

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


class BinaryWideResNet(nn.Module):
    def __init__(
        self, 
        depth: int, 
        widen_factor: int, 
        dropout_rate: float, 
        n_class: int
    ) -> None:
        super(BinaryWideResNet, self).__init__()

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
        return out.view(-1)

class BinaryEfficientNet_v1(nn.Module):
    def __init__(self, efficientnet_type):
        super(BinaryEfficientNet_v1, self).__init__()

        assert efficientnet_type in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

        self.models_dict = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7
        }
        self.model = self.models_dict[efficientnet_type](weights=None)  # No pretraining
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryTrainedEfficientNet_v1(nn.Module):
    def __init__(self, efficientnet_type):
        super(BinaryTrainedEfficientNet_v1, self).__init__()

        assert efficientnet_type in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

        self.models_dict = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7
        }
        self.model = self.models_dict[efficientnet_type](weights='DEFAULT')  # pre-trained
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryMobileNet(nn.Module):
    def __init__(self, mobilenet_type):
        super(BinaryMobileNet, self).__init__()

        assert mobilenet_type in ['mobilenet_v3_small','mobilenet_v3_large','mobilenet_v2']

        self.models_dict = {
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'mobilenet_v2': models.mobilenet_v2
        }
        self.model = self.models_dict[mobilenet_type](weights=None)  # No pretraining
        self.model.classifier[-1] = self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                          out_features=1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)
    
class BinaryTrainedMobileNet(nn.Module):
    def __init__(self, mobilenet_type):
        super(BinaryTrainedMobileNet, self).__init__()

        assert mobilenet_type in ['mobilenet_v3_small','mobilenet_v3_large','mobilenet_v2']

        self.models_dict = {
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'mobilenet_v2': models.mobilenet_v2
        }
        self.model = self.models_dict[mobilenet_type](weights='DEFAULT')
        for parameter in self.model.parameters():
            parameter.requires_grad = False  # No pretraining
        self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                          out_features=1)
          

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinarySqueezeNet(nn.Module):

    def __init__(self, squeezenet_type):
        super(BinarySqueezeNet, self).__init__()

        assert squeezenet_type in ['squeezenet1_0', 'squeezenet1_1']

        self.models_dict = {
            'squeezenet1_0': models.squeezenet1_0,
            'squeezenet1_1': models.squeezenet1_1
        }
        self.model = self.models_dict[squeezenet_type](weights=None)  # No pretraining
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryTrainedSqueezeNet(nn.Module):

    def __init__(self, squeezenet_type):
        super(BinaryTrainedSqueezeNet, self).__init__()

        assert squeezenet_type in ['squeezenet1_0', 'squeezenet1_1']

        self.models_dict = {
            'squeezenet1_0': models.squeezenet1_0,
            'squeezenet1_1': models.squeezenet1_1
        }
        self.model = self.models_dict[squeezenet_type](weights='DEFAULT')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryTrainedVGG(nn.Module):
    def __init__(self, vgg_type):
        super(BinaryTrainedVGG, self).__init__()
        
        assert vgg_type in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        
        self.model_dict = {
            'vgg11': models.vgg11,
            'vgg11_bn': models.vgg11_bn,
            'vgg13': models.vgg13,
            'vgg13_bn': models.vgg13_bn,
            'vgg16': models.vgg16,
            'vgg16_bn': models.vgg16_bn,
            'vgg19': models.vgg19,
            'vgg19_bn': models.vgg19_bn
        }

        self.model = self.model_dict[vgg_type](weights='DEFAULT')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.classifier[-1] = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryVGG(nn.Module):
    def __init__(self, vgg_type):
        super(BinaryVGG, self).__init__()
        
        assert vgg_type in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        
        self.model_dict = {
            'vgg11': models.vgg11,
            'vgg11_bn': models.vgg11_bn,
            'vgg13': models.vgg13,
            'vgg13_bn': models.vgg13_bn,
            'vgg16': models.vgg16,
            'vgg16_bn': models.vgg16_bn,
            'vgg19': models.vgg19,
            'vgg19_bn': models.vgg19_bn
        }

        self.model = self.model_dict[vgg_type](weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[6].in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryAlexNet(nn.Module):
    def __init__(self):
        super(BinaryAlexNet, self).__init__()
        
        self.model = models.alexnet(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryTrainedAlexNet(nn.Module):
    def __init__(self):
        super(BinaryTrainedAlexNet, self).__init__()
        
        self.model = models.alexnet(weights='DEFAULT')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryShuffleNet(nn.Module):
    def __init__(self, shuffle_type):
        super(BinaryShuffleNet, self).__init__()
        assert shuffle_type in ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 
                             'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
        self.model_dict = {
            'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
            'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
            'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
            'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0
        }
        self.model = self.model_dict[shuffle_type](weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
    def forward(self, x):
        x = self.model(x)
        return x.view(-1)

class BinaryTrainedShuffleNet(nn.Module):
    def __init__(self, shuffle_type):
        super(BinaryTrainedShuffleNet, self).__init__()
        assert shuffle_type in ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 
                             'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
        self.model_dict = {
            'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
            'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
            'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
            'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0
        }
        self.model = self.model_dict[shuffle_type](weights='DEFAULT')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
    def forward(self, x):
        x = self.model(x)
        return x.view(-1)


class BinarySmallCNN(nn.Module):
    def __init__(self):
        super(BinarySmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 16 x 112 x 112
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 32 x 56 x 56
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 64 x 28 x 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  # Correct dimension for 28x28 feature maps
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1)


# Define the CNN architecture
class BinaryLightweightCNN(nn.Module):
    def __init__(self):
        super(BinaryLightweightCNN, self).__init__()
        
        # Block 1: 3 -> 32 channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # No activation - raw logits
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x.view(-1)
