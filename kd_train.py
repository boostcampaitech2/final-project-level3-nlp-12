import torch
import wandb
import random
import argparse
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import data_loader.kd_data_loaders as module_data
from parse_config import ConfigParser
from trainer import KnowDistTrainer
from utils import prepare_device 
from transformers import AutoTokenizer
from data_loader.kd_data_loaders import KhsDataLoader


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
    wandb.init(project='#TODO', entity='#TODO', config=config)

    # build model architecture, then print to console
    student_model = config.init_obj('model', module_arch)
    teacher_model = config.init_obj('teacher_model', module_arch)
    
    # build tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['student']['type'])
    teacher_tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['teacher']['type'])

    # build train and valid dataloader
    dataloader = KhsDataLoader(
        student_tokenizer,
        teacher_tokenizer,
        max_length=config['data_loader']['args']['max_length']
    )
    train_data_loader = dataloader.get_dataloader(
        name='train',
        data_dir=config['data_loader']['args']['data_dir'], 
        data_files=config['data_loader']['data_files'],
        batch_size=config['data_loader']['args']['batch_size']
    )
    valid_data_loader = dataloader.get_dataloader(
        name='valid',
        data_dir=config['data_loader']['args']['data_dir'], 
        data_files=config['data_loader']['data_files'],
        batch_size=config['data_loader']['args']['batch_size']
    )

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    trainable_params = [
        {
            'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config['optimizer']['weight_decay']
        },
        {
            'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = KnowDistTrainer(
        student_model,
        teacher_model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    config = ConfigParser.from_args(args)
    main(config)
