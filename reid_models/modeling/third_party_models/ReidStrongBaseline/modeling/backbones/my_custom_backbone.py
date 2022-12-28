import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

# vgg19


class VGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.backbone = create_feature_extractor(vgg19, return_nodes=["features"])

    def forward(self, x):
        return self.backbone(x)["features"]


def vgg19():
    return VGG19()


# densenet121


class DENSENET121(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.backbone = create_feature_extractor(densenet121, return_nodes=["features"])

    def forward(self, x):
        return self.backbone(x)["features"]


def densenet121():
    return DENSENET121()


# shufflenet_v2


class SHUFFLENET_V2_X1_0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shufflenetv2 = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        self.backbone = create_feature_extractor(shufflenetv2, return_nodes=["conv5"])

    def forward(self, x):
        return self.backbone(x)["conv5"]


def shufflenetv2():
    return SHUFFLENET_V2_X1_0()


# mobilenetv2


class MOBILENET_V2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        self.backbone = create_feature_extractor(mobilenet, return_nodes=["features"])

    def forward(self, x):
        return self.backbone(x)["features"]


def mobilenetv2():
    return MOBILENET_V2()


# inceptionv3


class INCEPTION_V3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inceptionv3 = torchvision.models.inception_v3(pretrained=True)
        self.backbone = create_feature_extractor(inceptionv3, return_nodes=["Mixed_7c"])

    def forward(self, x):
        return self.backbone(x)["Mixed_7c"]


def inceptionv3():
    return INCEPTION_V3()


class EFFICIENTNET_B0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        efficientnet_b0 = torchvision.models.efficientnet_b0(pretrained=True)
        self.backbone = create_feature_extractor(
            efficientnet_b0, return_nodes=["features"]
        )

    def forward(self, x):
        return self.backbone(x)["features"]


def efficientnet_b0():
    return EFFICIENTNET_B0()


class RegNet_X_1_6GF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        regnet_x_1_6gf = torchvision.models.regnet_x_1_6gf(pretrained=True)
        self.backbone = create_feature_extractor(
            regnet_x_1_6gf, return_nodes=["trunk_output"]
        )

    def forward(self, x):
        return self.backbone(x)["trunk_output"]


def regnet_x_1_6gf():
    return RegNet_X_1_6GF()


class CONVNEXT_TINY(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        convnext_tiny = torchvision.models.convnext_tiny(pretrained=True)
        self.backbone = create_feature_extractor(
            convnext_tiny, return_nodes=["features"]
        )

    def forward(self, x):
        return self.backbone(x)["features"]


def convnext_tiny():
    return CONVNEXT_TINY()


class VIT_B_16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        self.backbone = create_feature_extractor(vit_b_16, return_nodes=["encoder"])

    def forward(self, x):
        return self.backbone(x)["encoder"]


def vit_b_16():
    return VIT_B_16()


# m = vit_b_16()
# x = m(torch.randn(1, 3, 224, 224))
# pass
