import torch.nn as nn
import torchvision.models as models

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
        self.resnet = self.models_dict[resnet_type](weights=None)  # No pretraining
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
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Change FC layer to 1 output

    def forward(self, x):
        x = self.resnet(x)
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
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=self.model.classifier[0].in_features, out_features=self.model.classifier[0].out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=self.model.classifier[3].in_features, out_features=1)
        )  if mobilenet_type in ['mobilenet_v3_small','mobilenet_v3_large'] else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=1)
        )

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
        self.model.classifier[6] = nn.Linear(4096, 1)

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
        self.model.classifier[6] = nn.Linear(4096, 1)

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