"""DWConv module, generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import math
# pylint: disable=useless-super-delegation
from typing import Union

import torch
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import Activation, autopad


class DWConv(nn.Module):
    """Depthwise convolution with batch normalization and activation."""

    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Depthwise convolution with batch normalization and activation.

        Args:
            in_channel: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        # error: Argument "padding" to "Conv2d" has incompatible type "Union[int, List[int]]";
        # expected "Union[int, Tuple[int, int]]"
        self.conv = nn.Conv2d(
            in_channel,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding),  # type: ignore
            groups=math.gcd(in_channel, out_channels),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.act(self.bn(self.conv(x)))

    def fusefoward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse forward."""
        return self.act(self.conv(x))


class DWConvGenerator(GeneratorAbstract):
    """Depth-wise convolution generator for parsing module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        if repeat > 1:
            stride = 1
            # Important!: stride only applies at the end of the repeat.
            if len(args) > 2:
                stride = args[3]
                args[3] = 1

            module = []
            for i in range(repeat):
                if len(args) > 1 and stride > 1 and i == repeat - 1:
                    args[3] = stride

                module.append(self.base_module(*args))
                args[0] = self.out_channel
        else:
            module = self.base_module(*args)

        return self._get_module(module)
