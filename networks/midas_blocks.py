import torch

import torch.nn.functional as F
import torch.nn as nn


def _make_encoder(features, use_pretrained):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x


class FeatLinearRefine(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.refine4 = FeatLinearRefineBlock(features)
        self.refine3 = FeatLinearRefineBlock(features)
        self.refine2 = FeatLinearRefineBlock(features)
        self.refine1 = FeatLinearRefineBlock(features)
        self.networks = [self.refine1, self.refine2,
                         self.refine3, self.refine4]

    def forward(self, feat_list):
        output_scale = []
        output_bias = []
        for x in range(len(feat_list)):
            scale, bias = self.networks[x](feat_list[x])
            output_scale.append(scale)
            output_bias.append(bias)
        return output_scale, output_bias

    def update_feat(self, feat_list, scales, biases):
        output_feat_list = []
        for x in range(len(feat_list)):
            output_feat_list.append(
                feat_list[x] * scales[x] + biases[x]
            )
        return output_feat_list


class FeatLinearRefineBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.res_conv_1 = ResidualConvUnit(in_dim)
        self.mid_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_weight = nn.Conv2d(
            in_dim//2, 1, kernel_size=3, stride=1, padding=1)
        self.output_bias = nn.Conv2d(
            in_dim//2, 1, kernel_size=3, stride=1, padding=1)
        self.output_weight.weight.data.normal_(0, 1e-4)
        self.output_weight.bias.data.fill_(1)
        self.output_bias.weight.data.normal_(0, 1e-4)
        self.output_bias.bias.data.fill_(0)

    def forward(self, x):
        x = self.res_conv_1(x)
        x = self.mid_conv(x)
        scale = self.output_weight(x)
        bias = self.output_bias(x)
        return scale, bias


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        y = F.relu(x, inplace=False)
        out = self.conv1(y)
        out = self.relu(out)
        out = self.conv2(out)

        return out + y


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output
