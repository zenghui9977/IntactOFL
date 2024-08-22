import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torch.nn.functional import avg_pool2d
import torch.nn.functional as F
import torch


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=num_classes)

    def feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)        

        return x

    def forward(self, x, get_feature=False):
        feature = self.feature(x)
        x = self.model.fc(feature)

        if get_feature:
            return x, feature
        else:
            return x

class ResNet34(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(ResNet34, self).__init__()
        self.model = resnet34(num_classes=num_classes)

    def feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)        

        return x

    def forward(self, x, get_feature=False):
        feature = self.feature(x)
        x = self.model.fc(feature)

        if get_feature:
            return x, feature
        else:
            return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(ResNet50, self).__init__()
        self.model = resnet50(num_classes=num_classes)


    def feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)        

        return x

    def forward(self, x, get_feature=False):
        feature = self.feature(x)
        x = self.model.fc(feature)

        if get_feature:
            return x, feature
        else:
            return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x, get_feature=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc2(x))
        x = self.fc3(z)

        if get_feature:
            return x, z
        else:
            return x
        
class CNNCifar(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(channels, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)


    def forward(self, x, get_feature=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        z = x.view(-1, 128 * 4 * 4)
        x = self.fc1(z)

        if get_feature:
            return x, z
        else:
            return x


class CNNCifar100(nn.Module):
    def __init__(self, num_classes=100, channels=3):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(channels, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)


    def forward(self, x, get_feature=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        z = x.view(-1, 128 * 4 * 4)
        x = self.fc1(z)

        if get_feature:
            return x, z
        else:
            return x

supported_models = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'lenet': LeNet,
    'cnn_cifar10': CNNCifar,
    'cnn_cifar100': CNNCifar100,
}


def get_model(model_name, num_classes=10, channels=3):
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model: {model_name}")
    model_class = supported_models[model_name]
    return model_class(num_classes, channels)

