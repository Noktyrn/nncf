import torch
from torchvision.models.vgg import cfgs, vgg19, _vgg, VGG
from typing import Any

cfgs['E1'] = cfgs['E'][:2] + cfgs['E'][3:5] + cfgs['E'][6:]

def vgg19_tiny(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E1', False, pretrained, progress, **kwargs)
