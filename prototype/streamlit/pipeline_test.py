from transformers import pipeline, AutoTokenizer
import torch
from utils.util import read_json
import argparse
import pprint
import time


def main(config):
    device = 0 if torch.cuda.is_available() else -1

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
        device=device
    )

    for i in range(config.num):
        text = input(f"문장을 입력하세요 {i+1} / {config.num}: ")

        start_time = time.time()
        result = pipe(text)
        end_time = time.time()
        
        print(f'inference time: {end_time - start_time}')
        pprint.pprint(result)
        

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
        default=3,
        type=int,
        help="How many times will you check the sentence? (default: 1)",
    )
    args = parser.parse_args()
    main(args)
