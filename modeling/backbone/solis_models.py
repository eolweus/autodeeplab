#!/usr/bin/env python3
from torchvision import models
from torchvision.models.segmentation import deeplabv3
# from torchvision.models import ResNet50
from torchvision.models._utils import IntermediateLayerGetter
import auto_deeplab as adl
import cell_level_search

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


# def resnet50(num_channels: int = 12, num_classes: int = 2):
#     model = models.resnet50()

#     bias = False if model.conv1.bias is None else True
#     model.conv1 = nn.Conv2d(
#         in_channels=num_channels,
#         out_channels=model.conv1.out_channels,
#         kernel_size=model.conv1.kernel_size,
#         stride=model.conv1.stride,
#         padding=model.conv1.padding,
#         bias=bias)

#     bias = False if model.fc is None else True
#     model.fc = nn.Linear(
#         in_features=model.fc.in_features,
#         out_features=num_classes,
#         bias=bias)

#     return model


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


# TODO: fix this
def autodeeplab50(num_bands: int, backbone_module: dict, num_classes: int, num_layers=12, criterion=None, filter_multiplier=8, block_multiplier=5, step=5, cell=cell_level_search.Cell):

    # Replace stride with dilation to make the model fit for use as backbone (important!)
    backbone = models.resnet50(
        replace_stride_with_dilation=(False, True, True))

    # Modify first layer of backbone to accept specified number of bands
    bias = False if backbone.conv1.bias is None else True
    backbone.conv1 = nn.Conv2d(
        in_channels=num_bands,
        out_channels=backbone.conv1.out_channels,
        kernel_size=backbone.conv1.kernel_size,
        stride=backbone.conv1.stride,
        padding=backbone.conv1.padding,
        bias=bias)

    # Modify last layer of backbone to output specified number of classes
    bias = False if backbone.fc.bias is None else True
    backbone.fc = nn.Linear(
        in_features=backbone.fc.in_features,
        out_features=num_classes,
        bias=bias)

    if backbone_module:
        backbone.load_state_dict(backbone_module['model_state_dict'])

    backbone = IntermediateLayerGetter(
        backbone, return_layers={"layer4": "out"})
    # TODO: fix this
    classifier = adl.AutoDeeplab(num_classes, num_layers=num_layers, criterion=criterion,
                                 filter_multiplier=filter_multiplier, block_multiplier=block_multiplier, step=step, cell=cell)

    return deeplabv3.DeepLabV3(backbone, classifier)


if __name__ == "__main__":
    # Code for removing the last layer of the resnet50 backbone
    # backbone = resnet50()
    # backbone_weights = backbone.state_dict()
    # del backbone_weights["fc.weight"]
    # del backbone_weights["fc.bias"]

    # model = deeplabv3_resnet50()
    # model.backbone.load_state_dict(backbone_weights)

    # print(model)
    pass
