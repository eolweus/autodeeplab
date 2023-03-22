#!/usr/bin/env python3
from torchvision import models

import torch.nn as nn


def resnet18(num_channels: int = 12, num_classes: int = 2):
    model = models.resnet18()
        
    bias = False if model.conv1.bias is None else True
    model.conv1 = nn.Conv2d(
        in_channels=num_channels, 
        out_channels=model.conv1.out_channels, 
        kernel_size=model.conv1.kernel_size, 
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=bias)

    bias = False if model.fc is None else True
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
        bias=bias)

    return model


def resnet50(num_channels: int = 12, num_classes: int = 2):
    model = models.resnet50()
        
    bias = False if model.conv1.bias is None else True
    model.conv1 = nn.Conv2d(
        in_channels=num_channels, 
        out_channels=model.conv1.out_channels, 
        kernel_size=model.conv1.kernel_size, 
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=bias)

    bias = False if model.fc is None else True
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
        bias=bias)

    return model


def deeplabv3_resnet50(num_channels: int = 12, num_classes: int = 2):
    model = models.segmentation.deeplabv3_resnet50(
        num_classes=num_classes,
        weights_backbone=None)

    bias = False if model.backbone.conv1.bias is None else True
    model.backbone.conv1 = nn.Conv2d(
        in_channels=num_channels, 
        out_channels=model.backbone.conv1.out_channels, 
        kernel_size=model.backbone.conv1.kernel_size, 
        stride=model.backbone.conv1.stride,
        padding=model.backbone.conv1.padding,
        bias=bias)

    return model


if __name__ == "__main__":
    backbone = resnet50()
    backbone_weights = backbone.state_dict()
    del backbone_weights["fc.weight"]
    del backbone_weights["fc.bias"]

    model = deeplabv3_resnet50()
    model.backbone.load_state_dict(backbone_weights)

    print(model)