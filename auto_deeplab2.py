from typing import Optional
import torch.nn as nn
import cell_level_search
from genotypes import PRIMITIVES
import torch.nn.functional as F
from operations import *
from decoding_formulas import Decoder
from torchvision import models

from torchvision.models._utils import IntermediateLayerGetter


class AutoDeeplab(nn.Module):
    def __init__(self, num_classes, num_layers, criterion=None, filter_multiplier=8, block_multiplier=5, step=5, cell=cell_level_search.Cell, num_bands: int = 3, backbone_module: Optional[dict] = None):
        super(AutoDeeplab, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._initialize_alphas_betas()
        self.num_bands = num_bands
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)

        if backbone_module:
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

            backbone.load_state_dict(backbone_module['model_state_dict'])

            backbone = IntermediateLayerGetter(
                backbone, return_layers={"layer4": "out"})

            self.backbone = backbone

            self.stem2 = nn.Sequential(
                nn.Conv2d(2048,
                          f_initial * self._block_multiplier, 3, stride=2, padding=1),
                nn.BatchNorm2d(f_initial * self._block_multiplier),
                nn.ReLU()
            )
        else:
            self.backbone = None

        self.stem0 = nn.Sequential(
            nn.Conv2d(self.num_bands, half_f_initial * self._block_multiplier,
                      3, stride=2, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._block_multiplier,
                      half_f_initial * self._block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )

        # handle the case where solis backbone is used
        if not backbone_module:
            self.stem2 = nn.Sequential(
                nn.Conv2d(half_f_initial * self._block_multiplier,
                          f_initial * self._block_multiplier, 3, stride=2, padding=1),
                nn.BatchNorm2d(f_initial * self._block_multiplier),
                nn.ReLU()
            )

        # intitial_fm = C_initial
        for i in range(self._num_layers):

            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier *
                             4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier *
                             4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.aspp_4 = nn.Sequential(
            ASPP(self._filter_multiplier * self._block_multiplier,
                 self._num_classes, 24, 24)  # 96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential(
            ASPP(self._filter_multiplier * 2 * self._block_multiplier,
                 self._num_classes, 12, 12)  # 96 / 8
        )
        self.aspp_16 = nn.Sequential(
            ASPP(self._filter_multiplier * 4 * self._block_multiplier,
                 self._num_classes, 6, 6)  # 96 / 16
        )
        self.aspp_32 = nn.Sequential(
            ASPP(self._filter_multiplier * 8 * self._block_multiplier,
                 self._num_classes, 3, 3)  # 96 / 32
        )

    def forward(self, x):
        # Check the chatgpt code above "benefits of avocado" for a better way to do this
        if self.backbone is not None:
            with torch.no_grad():
                # ResNet backbone returns a dict with keys 'out' and 'aux'
                # 'out' is the output of the last layer of the backbone
                # 'aux' is the output of the layer before the last layer
                # We only need 'out' for the decoder
                temp = self.backbone(x)['out']
                print(temp.shape)
            level_4_curr = self.stem2(temp)
        else:
            temp = self.stem0(x)
            temp = self.stem1(temp)
            level_4_curr = self.stem2(temp)
        # Solis

        count = 0
        normalized_betas = torch.randn(self._num_layers, 4, 3).cuda()
        # Softmax on alphas and betas
        if torch.cuda.device_count() > 1:
            print('1')
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(
                self.alphas.to(device=img_device), dim=-1)

            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2 / 3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(
                        self.betas[layer][1].to(device=img_device), dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(
                        self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(
                        self.betas[layer][2].to(device=img_device), dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(
                        self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(
                        self.betas[layer][2].to(device=img_device), dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(
                        self.betas[layer][3][:1].to(device=img_device), dim=-1) * (2 / 3)

        else:
            normalized_alphas = F.softmax(self.alphas, dim=-1)

            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:], dim=-1) * (2 / 3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:], dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(
                        self.betas[layer][1], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:], dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(
                        self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(
                        self.betas[layer][2], dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(
                        self.betas[layer][0][1:], dim=-1) * (2 / 3)
                    normalized_betas[layer][1] = F.softmax(
                        self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(
                        self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(
                        self.betas[layer][3][:2], dim=-1) * (2 / 3)

        # instantiate the level 4, 8, 16, 32 cells
        level_8_curr = None
        level_16_curr = None
        level_32_curr = None

        for layer in range(self._num_layers):

            level_4_prev = level_4_curr
            level_8_prev = level_8_curr
            level_16_prev = level_16_curr
            level_32_prev = level_32_curr

            if layer == 0:
                level_4_new, = self.cells[count](
                    None, None, level_4_curr, None, normalized_alphas)
                count += 1
                level_8_new, = self.cells[count](
                    None, level_4_curr, None, None, normalized_alphas)
                count += 1

                level_4_curr = normalized_betas[layer][0][1] * level_4_new
                level_8_curr = normalized_betas[layer][0][2] * level_8_new

            elif layer == 1:
                level4_new_1, level4_new_2 = self.cells[count](
                    level_4_prev, None, level_4_curr, level_8_curr, normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * \
                    level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2 = self.cells[count](
                    None, level_4_curr, level_8_curr, None, normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * \
                    level8_new_1 + normalized_betas[layer][1][2] * level8_new_2

                level16_new, = self.cells[count](
                    None, level_8_curr, None, None, normalized_alphas)
                level16_new = normalized_betas[layer][1][2] * level16_new
                count += 1

                level_4_curr = level4_new
                level_8_curr = level8_new
                level_16_curr = level16_new

            elif layer == 2:
                level4_new_1, level4_new_2 = self.cells[count](level_4_prev,
                                                               None,
                                                               level_4_curr,
                                                               level_8_curr,
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * \
                    level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](level_8_prev,
                                                                             level_4_curr,
                                                                             level_8_curr,
                                                                             level_16_curr,
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](None,
                                                                 level_8_curr,
                                                                 level_16_curr,
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * \
                    level16_new_1 + \
                    normalized_betas[layer][2][1] * level16_new_2

                level32_new, = self.cells[count](None,
                                                 level_16_curr,
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level32_new = normalized_betas[layer][2][2] * level32_new
                count += 1

                level_4_curr = level4_new
                level_8_curr = level8_new
                level_16_curr = level16_new
                level_32_curr = level32_new

            elif layer == 3:
                level4_new_1, level4_new_2 = self.cells[count](level_4_prev,
                                                               None,
                                                               level_4_curr,
                                                               level_8_curr,
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * \
                    level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](level_8_prev,
                                                                             level_4_curr,
                                                                             level_8_curr,
                                                                             level_16_curr,
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](level_16_prev,
                                                                                level_8_curr,
                                                                                level_16_curr,
                                                                                level_32_curr,
                                                                                normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][
                    0] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](None,
                                                                 level_16_curr,
                                                                 level_32_curr,
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level32_new = normalized_betas[layer][2][2] * \
                    level32_new_1 + \
                    normalized_betas[layer][3][1] * level32_new_2

                level_4_curr = level4_new
                level_8_curr = level8_new
                level_16_curr = level16_new
                level_32_curr = level32_new

            else:
                level4_new_1, level4_new_2 = self.cells[count](level_4_prev,
                                                               None,
                                                               level_4_curr,
                                                               level_8_curr,
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * \
                    level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](level_8_prev,
                                                                             level_4_curr,
                                                                             level_8_curr,
                                                                             level_16_curr,
                                                                             normalized_alphas)
                count += 1

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](level_16_prev,
                                                                                level_8_curr,
                                                                                level_16_curr,
                                                                                level_32_curr,
                                                                                normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][
                    0] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](level_32_prev,
                                                                 level_16_curr,
                                                                 level_32_curr,
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level32_new = normalized_betas[layer][2][2] * \
                    level32_new_1 + \
                    normalized_betas[layer][3][1] * level32_new_2

                level_4_curr = level4_new
                level_8_curr = level8_new
                level_16_curr = level16_new
                level_32_curr = level32_new

        aspp_result_4 = self.aspp_4(level_4_curr)
        aspp_result_8 = self.aspp_8(level_8_curr)
        aspp_result_16 = self.aspp_16(level_16_curr)
        aspp_result_32 = self.aspp_32(level_32_curr)
        upsample = nn.Upsample(
            size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = upsample(aspp_result_4)
        aspp_result_8 = upsample(aspp_result_8)
        aspp_result_16 = upsample(aspp_result_16)
        aspp_result_32 = upsample(aspp_result_32)

        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32

        return sum_feature_map

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        # alphas = torch.tensor(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        alphas = (1e-3 * torch.randn(k, num_ops)
                  ).clone().detach().requires_grad_(True)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)
                 ).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(
            self._arch_param_names, self._arch_parameters)]

    # def decode_viterbi(self):
    #     decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
    #     return decoder.viterbi_decode()

    # def decode_dfs(self):
    #     decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
    #     return decoder.dfs_decode()

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)


def main():
    model = AutoDeeplab(7, 12, None)
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    resultdfs = model.decode_dfs()
    resultviterbi = model.decode_viterbi()[0]

    print(resultviterbi)
    print(model.genotype())


if __name__ == '__main__':
    main()
