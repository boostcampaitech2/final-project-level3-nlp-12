"""Model parser and model.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
import yaml

from .modules import ModuleGenerator


class Model(nn.Module):
    """Base model class."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Parse model from the model config file.

        Args:
            cfg: yaml file path or dictionary type of the model.
            verbose: print the model parsing information.
        """
        super().__init__()
        self.model_parser = ModelParser(cfg=cfg, verbose=verbose)
        self.model = self.model_parser.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.forward_one(x)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward onetime."""

        return self.model(x)


class ModelParser:
    """Generate PyTorch model from the model yaml file."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Generate PyTorch model from the model yaml file.

        Args:
            cfg: model config file or dict values read from the model config file.
            verbose: print the parsed model information.
        """

        self.verbose = verbose
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        # self.in_channel = self.cfg["input_channel"]

        # self.depth_multiply = self.cfg["depth_multiple"]
        # self.width_multiply = self.cfg["width_multiple"]

        # error: Incompatible types in assignment (expression has type "Type[Any]",
        # variable has type "List[Union[int, str, float]]")
        self.model_cfg: List[Union[int, str, float]] = self.cfg["backbone"]  # type: ignore

        self.model = self._parse_model()

    def log(self, msg: str):
        """Log."""
        if self.verbose:
            print(msg)

    def _parse_model(self) -> nn.Sequential:
        """Parse model."""
        layers: List[nn.Module] = []
        log: str = (
            f"{'idx':>3} | {'n':>3} | {'params':>10} "
            f"| {'module':>15} | {'arguments':>20} | {'in_channel':>12} | {'out_channel':>13}"
        )
        self.log(log)
        self.log(len(log) * "-")  # type: ignore

        # in_channel = self.in_channel
        for i, (repeat, module, args) in enumerate(self.model_cfg):  # type: ignore
            repeat = (
                max(round(repeat * self.depth_multiply), 1) if repeat > 1 else repeat
            )

            module_generator = ModuleGenerator(module)(  # type: ignore
                *args,
                # width_multiply=self.width_multiply,
            )
            m = module_generator(repeat=repeat)

            layers.append(m)
            # in_channel = module_generator.out_channel

            log = (
                f"{i:3d} | {repeat:3d} | "
                f"{m.n_params:10,d} | {m.type:>15} | {str(args):>20} | "
                # f"{str(module_generator.in_channel):>12}"
                # f"{str(module_generator.out_channel):>13}"
            )

            self.log(log)

        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])
        # error: Incompatible return value type (got "Tuple[Sequential, List[int]]",
        # expected "Tuple[Module, List[Optional[int]]]")
        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d} "
            f"layers, {n_param:,d} parameters, {n_grad:,d} gradients"
        )

        return parsed_model
