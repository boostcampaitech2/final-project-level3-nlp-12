from transformers import pipeline, AutoTokenizer
import torch
from utils.util import read_json
import argparse
import pprint


def main(config):
    model_path = config.model_path
    config_path = config.config_path

    model = torch.load(model_path)
    model_config = read_json(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    pipe = pipeline(
        task="text-classification",
        config=model_config,
        model=model.model,
        tokenizer=tokenizer,
    )

    for i in range(config.num):
        pprint.pprint(pipe(input(f"문장을 입력하세요 {i+1} / {config.num}: ")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        default=None,
        type=str,
        help="saved model file (.pt) path (default: None)",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        default=None,
        type=str,
        help="saved model config file path (default: None)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="beomi/KcELECTRA-base",
        type=str,
        help="pretrained tokenizer name (default: beomi/KcELECTRA-base)",
    )
    parser.add_argument(
        "-n",
        "--num",
        default=1,
        type=int,
        help="How many times will you check the sentence? (default: 1)",
    )
    args = parser.parse_args()
    main(args)
