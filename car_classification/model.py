# import torchvision.models as models
# import torch.nn as nn
# import torch

# def get_model(num_classes):
#     model = models.resnet18(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)
#     return model

import torch
import torchvision.models as models
import torch.nn as nn

def get_model(num_classes):
    model = models.resnet50(pretrained=True)  # Use ResNet-50 or another model
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # Adjust final layer to match the number of classes
    return model
