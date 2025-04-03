import torch.nn as nn


class VGG16WithGradCAM(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        
        self.vgg = model
        self.features_conv = self.vgg.features[:30]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.classifier = self.vgg.classifier
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

class ResNetWithGradCAM(nn.Module):
    def __init__(self, model):
        super(ResNetWithGradCAM, self).__init__()
        self.model = model
        self.features_before = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.features = model.layer4
        self.pool = model.avgpool
        self.classifier = model.fc
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_before(x)
        x = self.features(x)
        # Register hook to capture gradients
        x.register_hook(self.activations_hook)
        out = self.pool(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.features_before(x)
        return self.features(x)

class SqueezeNetWithGradCAM(nn.Module):
    def __init__(self, model):
        super(SqueezeNetWithGradCAM, self).__init__()
        self.model = model
        self.features = self.model.features
        self.classifier = self.model.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)
