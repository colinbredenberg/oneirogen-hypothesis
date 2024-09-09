from __future__ import annotations

from collections import OrderedDict

import torch.nn.functional as F
from torch import nn

from beyond_backprop.networks.resnet import BasicBlock

from .pseudoinvert import pseudoinvert


class InvertedBasicBlock(nn.Module):
    """Implements basic block that mimics residual operation in inverted manner.

    Original residual forward pass:
    x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> + -> relu -> out
    |                                            |
    |------------ conv -> bn --------------------|

    Inverted residual forward pass:
    x -> relu -> bn2 -> conv2_t -> relu -> bn1 -> conv1_t -> + -> out
           |                                                 |
           |--------------- bn -> conv_t --------------------|
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.use_batchnorm = use_batchnorm

        # Create inverted layers
        self.conv1 = pseudoinvert(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.bn1 = pseudoinvert(nn.BatchNorm2d(planes) if self.use_batchnorm else nn.Identity())
        self.conv2 = pseudoinvert(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn2 = pseudoinvert(nn.BatchNorm2d(planes) if self.use_batchnorm else nn.Identity())

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                    conv=nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    bn=nn.BatchNorm2d(self.expansion * planes)
                    if self.use_batchnorm
                    else nn.Identity(),
                )
            )
        self.shortcut = pseudoinvert(self.shortcut)

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.conv2(self.bn2(x)))
        out = self.conv1(self.bn1(out))
        out += self.shortcut(x)
        return out


@pseudoinvert.register(BasicBlock)
def invert_basic(module: BasicBlock) -> InvertedBasicBlock:
    backward = InvertedBasicBlock(
        in_planes=module.in_planes,
        planes=module.planes,
        stride=module.stride,
        use_batchnorm=module.use_batchnorm,
    )
    return backward
