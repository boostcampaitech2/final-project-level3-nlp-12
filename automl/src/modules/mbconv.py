import math

import torch
import torch.nn as nn

from src.modules.base_generator import GeneratorAbstract


class MBConv(nn.Module):
    """MBConvBlock used in Efficientnet.

    Reference:
        https://github.com/narumiruna/efficientnet-pytorch/blob/master/efficientnet/models/efficientnet.py
    Note:
        Drop connect rate is disabled.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        expand_ratio,
        kernel_size,
        stride,
        reduction_ratio=4,
        drop_connect_rate=0.0,
    ):
        super(MBConv, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, in_planes // reduction_ratio)

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers.append(ConvBNReLU(in_planes, hidden_dim, 1))

        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride=stride,
                    groups=hidden_dim,
                ),
                # se
                SqueezeExcitation(hidden_dim, reduced_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        if self.drop_connect_rate >= 1.0:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding=0,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation layer used in MBConv."""

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvGenerator(GeneratorAbstract):
    """Bottleneck block generator."""

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
        """call method.

        InvertedResidualv3 args consists,
        repeat(=n), [c, t, s] // note original notation from paper is [t, c, n, s]
        """
        module = []
        t, c, s, k = self.args  # c is equivalent as self.out_channel
        inp, oup = self.in_channel, self.out_channel
        for i in range(repeat):
            stride = s if i == 0 else 1
            module.append(
                self.base_module(
                    in_planes=inp,
                    out_planes=oup,
                    expand_ratio=t,
                    stride=stride,
                    kernel_size=k,
                )
            )
            inp = oup
        return self._get_module(module)
