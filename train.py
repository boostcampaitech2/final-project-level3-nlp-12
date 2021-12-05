import os
import torch
import wandb
import random
import argparse
import collections
import numpy as np
import torch.nn as nn
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device 
from transformers import AutoTokenizer
from data_loader.data_loaders import KhsDataLoader
from datasets import load_dataset


def seed_everything(seed):
    """
    fix random seeds for reproducibility.
    Args:
        seed (int):
            seed number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    seed_everything(42)
    wandb.init(project='final-project', entity='jadon')

    # build model architecture and tokenizer
    model = config.init_obj('model', module_arch)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['args']['name'])
    
    # build train and valid dataloader
    dataloader = KhsDataLoader(
        tokenizer,
        max_length=config['data_loader']['args']['max_length']
    )
    train_dataloader = dataloader.get_dataloader(
        os.path.join(os.getcwd()[:], config['data_dir']['train']),
        batch_size=config['data_loader']['args']['batch_size']
    )
    valid_dataloader = dataloader.get_dataloader(
        os.path.join(os.getcwd()[:], config['data_dir']['valid']),
        batch_size=config['data_loader']['args']['batch_size']
    )

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    trainable_params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config['optimizer']['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=train_dataloader,
        valid_dataloader=valid_dataloader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()

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
