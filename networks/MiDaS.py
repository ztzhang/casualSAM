
from sys import path
from typing import ForwardRef
import torch
import torch.nn as nn
from .midas_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatLinearRefine


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


class MidasNet_mod(BaseModel):
    def __init__(self, path=None, features=256, non_negative=True, normalize_input=False, resize=None, freeze_backbone=False, mask_branch=False):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_mod, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

        if mask_branch:
            self.scratch.output_conv_mask = nn.Sequential(
                nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        self.mask_branch = mask_branch

        if normalize_input:
            self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
            self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.normalize_input = normalize_input
        self.resize = resize
        self.freeze_backbone = freeze_backbone

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def defrost(self):
        if self.freeze_backbone:
            for p in self.scratch.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """
        if self.normalize_input:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            x = x.permute([0, 2, 3, 1])
            x = (x - self.mean) / self.std
            x = x.permute([0, 3, 1, 2]).contiguous()

        orig_shape = x.shape[-2:]
        if self.resize is not None:
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode='bicubic', align_corners=True)

        if self.freeze_backbone:
            with torch.no_grad():
                layer_1 = self.pretrained.layer1(x)
                layer_2 = self.pretrained.layer2(layer_1)
                layer_3 = self.pretrained.layer3(layer_2)
                layer_4 = self.pretrained.layer4(layer_3)
        else:
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        out = torch.clamp(out, min=1e-2)

        out = 10000 / (out)

        if self.mask_branch:
            mask = self.scratch.output_conv_mask(path_1)

        else:
            mask = torch.zeros_like(out)

        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=orig_shape, mode='bicubic', align_corners=True)
            mask = torch.nn.functional.interpolate(
                mask, size=orig_shape, mode='bicubic', align_corners=True)
        return out, mask


class DistillBranch(nn.Module):
    def __init__(self, features, output_dim=32):
        super().__init__()
        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
            nn.Identity(),
        )

    def forward(self, feat_list):
        path_4 = self.scratch.refinenet4(feat_list[0])
        path_3 = self.scratch.refinenet3(path_4, feat_list[1])
        path_2 = self.scratch.refinenet2(path_3, feat_list[2])
        path_1 = self.scratch.refinenet1(path_2, feat_list[3])
        out = self.scratch.output_conv(path_1)
        return out


class DummyBranch(nn.Module):
    def __init__(self):
        super().__init__()


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def add_distillation_branch(self, output_dim=32):
        self.distill_branch = DistillBranch(self.features, output_dim)

    def forward_distillation(self, featlist):
        return self.distill_branch(featlist)

    def add_refine_branch(self, output_dim=32):
        self.refine_branch = nn.Module()

    def add_uncertainty_branch(self, output_channel=1):
        self.sigma_output = nn.Sequential(
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.feature_dim, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(32, output_channel, kernel_size=1, stride=1, padding=0),
            nn.ELU(True)
        )

        # self.sigma_output = DummyBranch()
        # self.sigma_output.refinenet4 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet3 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet2 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet1 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.output_conv = nn.Sequential(
        #     nn.Conv2d(self.feature_dim, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ELU(True),
        #     Interpolate(scale_factor=2, mode="bilinear"))

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(
                    m.weight.data, mean=0, std=1e-3)
                torch.nn.init.constant_(m.bias.data, 0.0)
        self.sigma_output.apply(init_func)
        # self.sigma_output.output_conv.apply(init_func)

    def __init__(self, path=None, features=256, non_negative=True, normalize_input=False, resize=None):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        self.feature_dim = features

        # self.sigma_output = nn.Sequential(
        #     nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ELU(True)
        # )

        if path:
            self.load(path)

        if normalize_input:
            self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
            self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.normalize_input = normalize_input
        self.resize = resize

    def forward_backbone(self, x):
        with torch.no_grad():
            if self.normalize_input:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
                x = x.permute([0, 2, 3, 1])
                x = (x - self.mean) / self.std
                x = x.permute([0, 3, 1, 2]).contiguous()

            self.orig_shape = x.shape[-2:]
            if self.resize is not None:
                x = torch.nn.functional.interpolate(
                    x, size=self.resize, mode='bilinear', align_corners=False)

            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
        return [layer_4_rn, layer_3_rn, layer_2_rn, layer_1_rn]

    def forward_refine(self, feat_list):
        path_4 = self.scratch.refinenet4(feat_list[0])
        path_3 = self.scratch.refinenet3(path_4, feat_list[1])
        path_2 = self.scratch.refinenet2(path_3, feat_list[2])
        path_1 = self.scratch.refinenet1(path_2, feat_list[3])

        out = self.scratch.output_conv(path_1)
        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=self.orig_shape, mode='bilinear', align_corners=False)
        return out

    def get_uncertainty_feature(self, x):
        if hasattr(self.sigma_output, "refinenet4"):
            return self.forward_backbone(x)
        else:
            feat_list = self.forward_backbone(x)
            # path_4 = self.scratch.refinenet4(feat_list[0])
            # path_3 = self.scratch.refinenet3(path_4, feat_list[1])
            # path_2 = self.scratch.refinenet2(path_3, feat_list[2])
            # path_1 = self.scratch.refinenet1(path_2, feat_list[3])
            return feat_list[1]

    def predict_uncertainty(self, uncertainty_feat):
        if hasattr(self.sigma_output, 'refinenet4'):
            sigma_4 = self.sigma_output.refinenet4(uncertainty_feat[0])
            sigma_3 = self.sigma_output.refinenet3(
                sigma_4, uncertainty_feat[1])
            sigma_2 = self.sigma_output.refinenet2(
                sigma_3, uncertainty_feat[2])
            sigma_1 = self.sigma_output.refinenet1(
                sigma_2, uncertainty_feat[3])
            uncertainty = self.sigma_output.output_conv(sigma_1)+1
        else:
            uncertainty = self.sigma_output(uncertainty_feat)+1
        if self.resize is not None:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=self.orig_shape, mode='bilinear', align_corners=False)
        return uncertainty

    def forward_refine_with_uncertainty(self, feat_list, no_depth_grad=False):
        if no_depth_grad:
            with torch.no_grad():
                path_4 = self.scratch.refinenet4(feat_list[0])
                path_3 = self.scratch.refinenet3(path_4, feat_list[1])
                path_2 = self.scratch.refinenet2(path_3, feat_list[2])
                path_1 = self.scratch.refinenet1(path_2, feat_list[3])
                out = self.scratch.output_conv(path_1)
        else:
            path_4 = self.scratch.refinenet4(feat_list[0].detach())
            path_3 = self.scratch.refinenet3(path_4, feat_list[1].detach())
            path_2 = self.scratch.refinenet2(path_3, feat_list[2].detach())
            path_1 = self.scratch.refinenet1(path_2, feat_list[3].detach())
            out = self.scratch.output_conv(path_1)
        if hasattr(self.sigma_output, 'refinenet4'):
            sigma_4 = self.sigma_output.refinenet4(feat_list[1].detach())
            sigma_3 = self.sigma_output.refinenet3(
                sigma_4, feat_list[1].detach())
            sigma_2 = self.sigma_output.refinenet2(
                sigma_3, feat_list[2].detach())
            sigma_1 = self.sigma_output.refinenet1(
                sigma_2, feat_list[3].detach())
            uncertainty = self.sigma_output.output_conv(sigma_1)+1
        else:
            uncertainty = self.sigma_output(feat_list[1])+1
        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=self.orig_shape, mode='bilinear', align_corners=False)
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=self.orig_shape, mode='bilinear', align_corners=False)
        return out, uncertainty

    def forward(self, x, inverse_depth=False, return_feat=False, freeze_backbone=False):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """
        if self.normalize_input:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            x = x.permute([0, 2, 3, 1])
            x = (x - self.mean) / self.std
            x = x.permute([0, 3, 1, 2]).contiguous()

        orig_shape = x.shape[-2:]
        if self.resize is not None:
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode='bilinear', align_corners=False)

        if freeze_backbone:
            with torch.no_grad():
                layer_1 = self.pretrained.layer1(x)
                layer_2 = self.pretrained.layer2(layer_1)
                layer_3 = self.pretrained.layer3(layer_2)
                layer_4 = self.pretrained.layer4(layer_3)

                layer_1_rn = self.scratch.layer1_rn(layer_1)
                layer_2_rn = self.scratch.layer2_rn(layer_2)
                layer_3_rn = self.scratch.layer3_rn(layer_3)
                layer_4_rn = self.scratch.layer4_rn(layer_4)
        else:
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        if not inverse_depth:
            out = torch.clamp(out, min=1e-2)
            out = 10000 / (out)
        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=orig_shape, mode='bilinear', align_corners=False)
        if return_feat:
            return [layer_1, layer_2, layer_3, layer_4, layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn, path_1, path_2, path_3, path_4, out]
        return out


def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        torch.nn.init.constant_(m.bias.data, 0.0)


class MidasNet_featopt(BaseModel):
    """Network for monocular depth estimation.
    """

    # def add_distillation_branch(self, output_dim=32):
    #     self.distill_branch = DistillBranch(self.features, output_dim)

    # def forward_distillation(self, featlist):
    #    return self.distill_branch(featlist)

    def add_refine_branch(self):
        self.refine_branch = FeatLinearRefine(self.feature_dim)

    def add_uncertainty_branch(self):
        self.sigma_output = nn.Sequential(
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.feature_dim, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ELU(True)
        )

        # self.sigma_output = DummyBranch()
        # self.sigma_output.refinenet4 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet3 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet2 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet1 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.output_conv = nn.Sequential(
        #     nn.Conv2d(self.feature_dim, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ELU(True),
        #     Interpolate(scale_factor=2, mode="bilinear"))

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(
                    m.weight.data, mean=0, std=1e-3)
                torch.nn.init.constant_(m.bias.data, 0.0)
        self.sigma_output.apply(init_func)
        # self.sigma_output.output_conv.apply(init_func)

    def __init__(self, path=None, features=256, non_negative=True, normalize_input=False, resize=None):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super().__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        self.feature_dim = features

        # self.sigma_output = nn.Sequential(
        #     nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ELU(True)
        # )

        if path:
            self.load(path)

        if normalize_input:
            self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
            self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.normalize_input = normalize_input
        self.resize = resize

    def forward_backbone(self, x):
        with torch.no_grad():
            if self.normalize_input:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
                x = x.permute([0, 2, 3, 1])
                x = (x - self.mean) / self.std
                x = x.permute([0, 3, 1, 2]).contiguous()

            self.orig_shape = x.shape[-2:]
            if self.resize is not None:
                x = torch.nn.functional.interpolate(
                    x, size=self.resize, mode='bilinear', align_corners=False)

            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
        return [layer_4_rn, layer_3_rn, layer_2_rn, layer_1_rn]

    def forward_refine(self, feat_list):
        scales, biases = self.refine_branch(feat_list)
        output_feat = self.refine_branch.update_feat(
            feat_list, [1. for s in scales], biases)

        path_4 = self.scratch.refinenet4(output_feat[0])
        path_3 = self.scratch.refinenet3(path_4, output_feat[1])
        path_2 = self.scratch.refinenet2(path_3, output_feat[2])
        path_1 = self.scratch.refinenet1(path_2, output_feat[3])

        out = self.scratch.output_conv(path_1)
        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=self.orig_shape, mode='bilinear', align_corners=False)
        return out

    def get_uncertainty_feature(self, x):
        if hasattr(self.sigma_output, "refinenet4"):
            return self.forward_backbone(x)
        else:
            feat_list = self.forward_backbone(x)
            # path_4 = self.scratch.refinenet4(feat_list[0])
            # path_3 = self.scratch.refinenet3(path_4, feat_list[1])
            # path_2 = self.scratch.refinenet2(path_3, feat_list[2])
            # path_1 = self.scratch.refinenet1(path_2, feat_list[3])
            return feat_list[1]

    def predict_uncertainty(self, uncertainty_feat):
        if hasattr(self.sigma_output, 'refinenet4'):
            sigma_4 = self.sigma_output.refinenet4(uncertainty_feat[0])
            sigma_3 = self.sigma_output.refinenet3(
                sigma_4, uncertainty_feat[1])
            sigma_2 = self.sigma_output.refinenet2(
                sigma_3, uncertainty_feat[2])
            sigma_1 = self.sigma_output.refinenet1(
                sigma_2, uncertainty_feat[3])
            uncertainty = self.sigma_output.output_conv(sigma_1)+1
        else:
            uncertainty = self.sigma_output(uncertainty_feat)+1
        if self.resize is not None:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=self.orig_shape, mode='bilinear', align_corners=False)
        return uncertainty

    def forward_refine_with_uncertainty(self, feat_list, no_depth_grad=False):
        if no_depth_grad:
            with torch.no_grad():
                disp = self.forward_refine(feat_list)
        else:
            disp = self.forward_refine(feat_list)
        if hasattr(self.sigma_output, 'refinenet4'):
            sigma_4 = self.sigma_output.refinenet4(feat_list[1].detach())
            sigma_3 = self.sigma_output.refinenet3(
                sigma_4, feat_list[1].detach())
            sigma_2 = self.sigma_output.refinenet2(
                sigma_3, feat_list[2].detach())
            sigma_1 = self.sigma_output.refinenet1(
                sigma_2, feat_list[3].detach())
            uncertainty = self.sigma_output.output_conv(sigma_1)+1
        else:
            uncertainty = self.sigma_output(feat_list[1])+1
        if self.resize is not None:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=self.orig_shape, mode='bilinear', align_corners=False)
        return disp, uncertainty

    def forward(self, x, inverse_depth=False, return_feat=False, freeze_backbone=False):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """
        if self.normalize_input:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            x = x.permute([0, 2, 3, 1])
            x = (x - self.mean) / self.std
            x = x.permute([0, 3, 1, 2]).contiguous()

        orig_shape = x.shape[-2:]
        if self.resize is not None:
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode='bilinear', align_corners=False)

        if freeze_backbone:
            with torch.no_grad():
                layer_1 = self.pretrained.layer1(x)
                layer_2 = self.pretrained.layer2(layer_1)
                layer_3 = self.pretrained.layer3(layer_2)
                layer_4 = self.pretrained.layer4(layer_3)

                layer_1_rn = self.scratch.layer1_rn(layer_1)
                layer_2_rn = self.scratch.layer2_rn(layer_2)
                layer_3_rn = self.scratch.layer3_rn(layer_3)
                layer_4_rn = self.scratch.layer4_rn(layer_4)
        else:
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        if not inverse_depth:
            out = torch.clamp(out, min=1e-2)
            out = 10000 / (out)
        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=orig_shape, mode='bilinear', align_corners=False)
        if return_feat:
            return [layer_1, layer_2, layer_3, layer_4, layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn, path_1, path_2, path_3, path_4, out]
        return out
