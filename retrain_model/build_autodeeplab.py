import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torchvision.models.segmentation import deeplabv3

from operations import NaiveBN, ABN
from retrain_model.aspp import ASPP
from retrain_model.decoder import Decoder
from retrain_model.new_model import get_default_arch, newModel, network_layer_to_space

from collections import OrderedDict


class Retrain_Autodeeplab(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        BatchNorm2d = ABN if args.use_ABN else NaiveBN
        if (not args.dist and args.use_ABN) or (args.dist and args.use_ABN and dist.get_rank() == 0):
            print("=> use ABN!")
        if args.net_arch is not None and args.cell_arch is not None:
            network_path = np.load(args.net_arch)
            cell_arch = np.load(args.cell_arch)
            network_arch = network_layer_to_space(network_path)
        else:
            network_arch, cell_arch, network_path = get_default_arch()
        self.encoder = newModel(network_arch, cell_arch, args.num_classes,
                                12, args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)
        self.aspp = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[-1]],
                         256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        # self.decoder = Decoder(args.num_classes, filter_multiplier=args.filter_multiplier * args.block_multiplier,
        #                        args=args, last_level=network_path[-1])

        # we use a filtermultiplier of 128 for the decoder, because we wish to use an input that is downscaled by 4
        # The only way to guarantee this is to use a filtermultiplier of 128 and take the output of the stem as the input
        self.decoder = Decoder(
            args.num_classes, filter_multiplier=128, args=args, last_level=network_path[-1])

    def forward(self, x):
        encoder_output, low_level_feature = self.encoder(x)
        high_level_feature = self.aspp(encoder_output)
        decoder_output = self.decoder(high_level_feature, low_level_feature)
        return nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(decoder_output)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
            + list(self.decoder.parameters()) \
            + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params


class Retrain_Autodeeplab2(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab2, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        BatchNorm2d = ABN if args.use_ABN else NaiveBN
        if (not args.dist and args.use_ABN) or (args.dist and args.use_ABN and dist.get_rank() == 0):
            print("=> use ABN!")
        if args.net_arch is not None and args.cell_arch is not None:
            network_path = np.load(args.net_arch)
            cell_arch = np.load(args.cell_arch)
            network_arch = network_layer_to_space(network_path)
        else:
            network_arch, cell_arch, network_path = get_default_arch()
        self.encoder = newModel(network_arch, cell_arch, args.num_classes,
                                12, args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)

        classifier = deeplabv3.DeepLabHead(
            args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[-1]], 2)
        self.model = DeepLabV3(self.encoder, classifier)

    def forward(self, x):
        return self.model(x)

    # TODO: this is not correct
    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
            + list(self.decoder.parameters()) \
            + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params


class DeepLabV3(nn.Module):

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]

        features, other = self.backbone(x)

        x = features
        x = self.classifier(x)
        result = F.interpolate(x, size=input_shape,
                               mode="bilinear", align_corners=False)

        return result
