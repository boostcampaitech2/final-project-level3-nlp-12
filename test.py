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

IDX_2_LABEL = {
    0: "none",
    1: "offensive",
    2: "hate"
}

def main(config):
    # load model and tokenizer architecture
    # model = torch.load(config['model']['type'])
    model = torch.load('/opt/ml/final-project-level3-nlp-12/saved/best/beomi/beep-KcELECTRA-base-hate/best_model.pt')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['args']['name'])

    # setup data_loader instances
    data_loader = getattr(module_data, 'KhsDataLoader')(
        tokenizer,
        max_length=config['data_loader']['args']['max_length']
    )
    data_loader = data_loader.get_dataloader(
        name='test',
        data_dir=config['data_dir'], 
        data_files=config['test_data_file'],
        batch_size=config['data_loader']['args']['batch_size']
    )

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
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
            outputs = model(inputs)
            
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0]
            
            _, preds = torch.max(logits, dim=1)
            
            output_pred.extend(preds.detach().cpu().numpy())
            
    df = pd.read_csv('/opt/ml/final-project-level3-nlp-12/data/korean-hate-speech/origin/test/test_hate_no_label.csv')
    df['label'] = output_pred
        
    df.to_csv(
        'data/result.csv',
        index=None
    )

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
