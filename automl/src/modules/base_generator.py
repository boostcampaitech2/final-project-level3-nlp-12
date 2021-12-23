"""Base Module Generator.

This module is responsible for GeneratorAbstract and ModuleGenerator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from abc import ABC, abstractmethod
from typing import List, Union

from torch import nn as nn

from src.utils.torch_utils import make_divisible


class GeneratorAbstract(ABC):
    """Abstract Module Generator."""

    CHANNEL_DIVISOR: int = 8

    def __init__(
        self,
        *args,
        from_idx: Union[int, List[int]] = -1,
    ):
        """Initialize module generator.

        Args:
            *args: Module arguments
            from_idx: Module input index
        """
        self.args = tuple(args)
        self.from_idx = from_idx

    @property
    def name(self) -> str:
        """Module name."""
        return self.__class__.__name__.replace("Generator", "")

    def _get_module(self, module: Union[nn.Module, List[nn.Module]]) -> nn.Module:
        """Get module from __call__ function."""
        if isinstance(module, list):
            module = nn.Sequential(*module)

        # error: Incompatible types in assignment (expression has type "Union[Tensor, Module, int]",
        # variable has type "Union[Tensor, Module]")
        # error: List comprehension has incompatible type List[int];
        # expected List[Union[Tensor, Module]]
        module.n_params = sum([x.numel() for x in module.parameters()])  # type: ignore
        # error: Cannot assign to a method
        module.type = self.name  # type: ignore

        return module

    # @classmethod
    # def _get_divisible_channel(cls, n_channel: int) -> int:
    #     """Get divisible channel by default divisor.

    #     Args:
    #         n_channel: number of channel.

    #     Returns:
    #         Ex) given {n_channel} is 52 and {GeneratorAbstract.CHANNEL_DIVISOR} is 8.,
    #             return channel is 56 since ceil(52/8) = 7 and 7*8 = 56
    #     """
    #     return make_divisible(n_channel, divisor=cls.CHANNEL_DIVISOR)

    # @property
    # @abstractmethod
    # def out_channel(self) -> int:
    #     """Out channel of the module."""

    @abstractmethod
    def __call__(self, repeat: int = 1):
        """Returns nn.Module component"""


class ModuleGenerator:
    """Module generator class."""

    def __init__(self, module_name: str):
        """Generate module based on the {module_name}

        Args:
            module_name: {module_name}Generator class must have been implemented.
        """
        self.module_name = module_name
        # self.in_channel = in_channel

    def __call__(self, *args, **kwargs):
        # replace getattr
        return getattr(
            __import__("src.modules", fromlist=[""]),
            f"{self.module_name}Generator",
        )(*args, **kwargs)
