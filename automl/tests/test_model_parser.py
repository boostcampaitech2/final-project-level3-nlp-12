"""Model parse test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import torch

from src.model import Model


class TestModelParser:
    """Test model parser."""

    # pylint: disable=no-self-use

    INPUT = torch.rand(8, 3, 256, 256)

    def test_mobilenetv3(self):
        """Test mobilenetv3 model."""
        model = Model(os.path.join("configs", "model", "mobilenetv3.yaml"))
        assert model(TestModelParser.INPUT).shape == torch.Size([8, 9])

    def test_example(self):
        """Test example model."""
        model = Model(os.path.join("configs", "model", "example.yaml"))
        assert model(TestModelParser.INPUT).shape == torch.Size([8, 9])


if __name__ == "__main__":
    test = TestModelParser()

    test.test_mobilenetv3()
    test.test_example()
