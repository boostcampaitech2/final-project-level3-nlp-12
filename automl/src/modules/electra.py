import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from transformers import AutoConfig, AutoModel
from src.modules.base_generator import GeneratorAbstract


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class Electra(BaseModel):
    def __init__(self, name="beomi/beep-KcELECTRA-base-hate"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs[0]


class ElectraGenerator(GeneratorAbstract):
    """Pretrained Electra block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method.
        Build module
        """
        module = []
        args = self.args
        for i in range(repeat):
            module.append(self.base_module(*args))
        return self._get_module(module)
