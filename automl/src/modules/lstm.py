import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
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


class Lstm(BaseModel):
    def __init__(self, name="rnn", xdim=28, hdim=256, ydim=3, n_layer=3, dropout=0):
        super(Lstm, self).__init__()
        self.name = name
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.n_layer = n_layer  # K
        self.dropout = dropout

        self.rnn = nn.LSTM(
            input_size=xdim,
            hidden_size=hdim,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout,
        )
        self.lin = nn.Linear(self.hdim, self.ydim)

    def forward(self, x):
        # Set initial hidden and cell states
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        # print(device)
        h0 = torch.zeros(self.n_layer, x.size(0), self.hdim).to(device)
        c0 = torch.zeros(self.n_layer, x.size(0), self.hdim).to(device)

        # RNN
        rnn_out, (hn, cn) = self.rnn(x, (h0, c0))
        # x:[N x L x Q] => rnn_out:[N x L x D]
        # Linear
        out = self.lin(rnn_out[:, -1, :]).view([-1, self.ydim])
        return out


class LstmGenerator(GeneratorAbstract):
    """Pretrained Bert block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method."""
        module = []
        args = self.args
        for i in range(repeat):
            module.append(self.base_module(*args))
        return self._get_module(module)
