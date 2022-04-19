from turtle import forward
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import midas_pretrain_path
from networks.midas_blocks import Interpolate
from .uncertainty_helper import get_hrnet_segmentation_model
from functools import partial
from networks.MiDaS import MidasNet

interpolate2D = partial(F.interpolate, mode='bilinear', align_corners=True)


def get_uncertainty_model(name):
    pass


class MidasUnceratintyModel(nn.Module):
    def __init__(self, output_shape=None, output_channel=1, constant_uncertainty=False) -> None:
        super().__init__()
        self.encoder_model = MidasNet(path=midas_pretrain_path,
                                      normalize_input=True, non_negative=True).eval()
        self.encoder_model.add_uncertainty_branch(
            output_channel=1)
        self.uncertainty_decoder = self.encoder_model.sigma_output
        if output_channel > 1:
            self.uncertainty_decoder.register_parameter(
                'scale', nn.Parameter(torch.ones([1])*-2))
        self.constant_uncertainty = constant_uncertainty

    def get_uncertainty_feat(self, imgs):
        self.encoder_model.eval()
        self.img_shape = imgs.shape[2:]
        with torch.no_grad():
            feats = self.encoder_model.forward_backbone(imgs)
        return feats

    def forward(self, feat):
        f = feat[1]
        B = f.shape[0]
        if self.constant_uncertainty:
            return torch.ones([B, 2, *self.img_shape]).to(f.device)*0.5
        uncertainty = self.uncertainty_decoder(f)+1
        if hasattr(self.uncertainty_decoder, 'scale'):
            # / (self.uncertainty_decoder.scale+1e-6)
            uncertainty_2 = uncertainty * \
                F.softplus(self.uncertainty_decoder.scale, beta=2)
            uncertainty = torch.cat([uncertainty, uncertainty_2], dim=1)
        return uncertainty


class ResNetUncertaintyModel():
    pass


class HRNetUncertaintyModel(nn.Module):
    def __init__(self, output_shape=None):
        super().__init__()
        self.encoder_model = get_hrnet_segmentation_model().eval()
        self.uncertainty_decoder = PixelDecoder(inchannels=720)
        self.output_shape = output_shape

    def get_uncertainty_feat(self, imgs):
        '''
        return features to be used by the uncertainty model. usually in the form of list of tensors with different sizes
        '''
        self.encoder_model.eval()
        with torch.no_grad():
            feats = self.encoder_model(imgs, feat_only=True)
        return feats

    def forward(self, feat):
        if type(feat) is list:
            outfeat = [feat[0]]
            h, w = feat[0].shape[2:]
            for i in range(1, len(feat)):
                outfeat.append(interpolate2D(feat[i], size=(h, w)))
            outfeat = torch.cat(outfeat, dim=1)
        else:
            outfeat = feat
        uncertainty = self.uncertainty_decoder(outfeat)
        if self.output_shape is not None:
            uncertainty = interpolate2D(uncertainty, size=self.output_shape)
        return uncertainty


class PixelDecoder(nn.Module):
    def __init__(self, inchannels=256, outchannels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(inchannels, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64,
                      kernel_size=3, stride=1, padding=1),
            # nn.InstanceNorm2d(inchannels//4),
            # nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32,
                      kernel_size=3, stride=1, padding=1),
            # nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, outchannels, kernel_size=1, stride=1, padding=0),
            nn.Softplus(beta=4, threshold=2)
        )

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(
                    m.weight.data, mean=0, std=1e-2)
                torch.nn.init.constant_(m.bias.data, 0.0)
        self.model.apply(init_func)
        nn.init.normal_(self.model[-2].weight, std=0.001, mean=0)
        nn.init.constant_(self.model[-2].bias, 0)

    def forward(self, feat):
        return self.model(feat)


##
#
#    # nn.InstanceNorm2d(256),
#         nn.ReLU(inplace=False),
#         nn.Conv2d(self.feature_dim, 128,
#                   kernel_size=3, stride=1, padding=1),
#         nn.InstanceNorm2d(128),
#         nn.ReLU(True),
#         Interpolate(scale_factor=2, mode="bilinear"),
#         nn.Conv2d(128, 128,
#                   kernel_size=3, stride=1, padding=1),
#         nn.InstanceNorm2d(128),
#         nn.ReLU(True),
#         Interpolate(scale_factor=2, mode="bilinear"),
#         nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#         nn.InstanceNorm2d(64),
#         nn.ReLU(True),
#         Interpolate(scale_factor=2, mode="bilinear"),
#         nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#         nn.InstanceNorm2d(32),
#         nn.ReLU(True),
#         Interpolate(scale_factor=2, mode="bilinear"),
#         nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#         nn.ELU(True)
