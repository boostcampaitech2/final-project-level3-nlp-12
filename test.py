import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from transformers import AutoTokenizer
from parse_config import ConfigParser
from datasets import load_dataset

IDX_2_LABEL = {
    0: "none",
    1: "offensive",
    2: "hate"
}

def main(config):
    # load model and tokenizer architecture
    model = torch.load(config['model']['type'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['type'])

    # setup data_loader instances
    data_loader = getattr(module_data, 'KhsDataLoader')(
        tokenizer,
        max_length=config['data_loader']['args']['max_length']
    )
    data_loader = data_loader.get_dataloader(
        name='test',
        data_dir=config['data_loader']['args']['data_dir'],
        data_files=config['data_loader']['test_data_file'],
        batch_size=config['data_loader']['args']['batch_size']
    )

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    scaler = (
        torch.cuda.amp.GradScaler() if config['fp16'] and device != torch.device("cpu") else None
    )
    
    model.eval()

    output_pred = []

    with torch.no_grad():
        for step, data in enumerate(tqdm(data_loader)):
            input_ids, token_type_ids, attention_mask = data
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0]
            
            _, preds = torch.max(logits, dim=1)
            
            output_pred.extend(preds.detach().cpu().numpy())
            
    dataset = load_dataset(config['data_loader']['args']['data_dir'], data_files=config['data_loader']['test_data_file'], use_auth_token=True)
    test_df = pd.DataFrame()
    test_df['comments'] = dataset['test']['comments']
    test_df['label'] = output_pred
    test_df.to_csv(
        'data/result.csv',
        index=None
    )

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
