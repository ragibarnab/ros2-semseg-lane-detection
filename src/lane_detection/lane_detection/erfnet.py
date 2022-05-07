# Copied and modified from Eromera/erfnet_pytorch,
# cardwing/Codes-for-Lane-Detection and
# jcdubron/scnn_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .non_bottleneck_1d import non_bottleneck_1d
from .encoder_decoder_lane_exist import EDLaneExist


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class Encoder(nn.Module):
    def __init__(self, num_classes, dropout_1=0.03, dropout_2=0.3):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, dropout_1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, dropout_2, 2))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 4))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 8))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
class ERFNet(nn.Module):
    def __init__(self,
                 num_classes,
                 dropout_1=0.3,
                 dropout_2=0.3):
        super().__init__()
        self.encoder = Encoder(num_classes=num_classes, dropout_1=dropout_1, dropout_2=dropout_2)
        self.decoder = Decoder(num_classes)
        self.lane_classifier = EDLaneExist(num_classes-1, flattened_size=4400, dropout=0.3, pool='max')


    def forward(self, x):
        out = OrderedDict()
        output = self.encoder(x)    # predict=False by default
        out['out'] = self.decoder.forward(output)
        if self.lane_classifier is not None:
            out['lane'] = self.lane_classifier(output)
        return out
