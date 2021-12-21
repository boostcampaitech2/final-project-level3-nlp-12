"""Unit test for model conversion to TorchScript.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""


import os

import torch

from src.model import Model
from src.utils.torch_utils import convert_model_to_torchscript


class TestModelConversion:
    """Test model conversion."""

    # pylint: disable=no-self-use

    INPUT1 = torch.rand(1, 3, 128, 128)
    INPUT2 = torch.rand(8, 3, 256, 256)
    SAVE_PATH = "tests/.test_model.ts"

    def _convert_evaluation(self, path: str) -> None:
        """Model conversion test."""
        model = Model(path)
        convert_model_to_torchscript(model, path=TestModelConversion.SAVE_PATH)

        ts_model = torch.jit.load(TestModelConversion.SAVE_PATH)

        out_tensor1 = ts_model(TestModelConversion.INPUT1)
        out_tensor2 = ts_model(TestModelConversion.INPUT2)

        os.remove(TestModelConversion.SAVE_PATH)
        assert out_tensor1.shape == torch.Size((1, 9))
        assert out_tensor2.shape == torch.Size((8, 9))

    def test_mobilenetv3(self):
        """Test convert mobilenetv3 model to TorchScript."""
        self._convert_evaluation(os.path.join("configs", "model", "mobilenetv3.yaml"))

    def test_example(self):
        """Test convert example model to TorchScript."""
        self._convert_evaluation(os.path.join("configs", "model", "example.yaml"))


if __name__ == "__main__":
    test = TestModelConversion()
    test.test_mobilenetv3()
    test.test_example()
