"""Module generator related to pooling operations.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
# pylint: disable=useless-super-delegation
from torch import nn

from src.modules.base_generator import GeneratorAbstract


class MaxPoolGenerator(GeneratorAbstract):
    """Max pooling module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channel

    @property
    def base_module(self) -> nn.Module:
        """Base module."""
        return getattr(nn, f"{self.name}2d")

    def __call__(self, repeat: int = 1):
        module = (
            [self.base_module(*self.args) for _ in range(repeat)]
            if repeat > 1
            else self.base_module(*self.args)
        )
        return self._get_module(module)


class AvgPoolGenerator(MaxPoolGenerator):
    """Average pooling module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    """Global average pooling module."""

    def __init__(self, output_size=1):
        """Initialize."""
        super().__init__(output_size=output_size)


class GlobalAvgPoolGenerator(GeneratorAbstract):
    """Global average pooling module generator."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__(*args, **kwargs)
        self.output_size = 1
        if len(args) > 1:
            self.output_size = args[1]

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        return self._get_module(GlobalAvgPool(self.output_size))
