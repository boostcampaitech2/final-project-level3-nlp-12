"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.invertedresidualv2 import (
    InvertedResidualv2,
    InvertedResidualv2Generator,
)
from src.modules.invertedresidualv3 import (
    InvertedResidualv3,
    InvertedResidualv3Generator,
)
from src.modules.linear import Linear, LinearGenerator
from src.modules.poolings import (
    AvgPoolGenerator,
    GlobalAvgPool,
    GlobalAvgPoolGenerator,
    MaxPoolGenerator,
)
from src.modules.bert import Bert, BertGenerator
from src.modules.electra import Electra, ElectraGenerator
from src.modules.lstm import Lstm, LstmGenerator
from src.modules.electra_lstm import ElectraWithLSTM, ElectraWithLSTMGenerator
from src.modules.bert_lstm import BertWithLSTM, BertWithLSTMGenerator

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Conv",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "InvertedResidualv2",
    "InvertedResidualv3",
    "BottleneckGenerator",
    "FixedConvGenerator",
    "ConvGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "InvertedResidualv2Generator",
    "InvertedResidualv3Generator",
    "Bert",
    "BertGenerator",
    "Electra",
    "ElectraGenerator" "Lstm",
    "LstmGenerator",
    "ElectraWithLSTM",
    "ElectraWithLSTMGenerator",
    "BertWithLSTM",
    "BertWithLSTMGenerator",
]
