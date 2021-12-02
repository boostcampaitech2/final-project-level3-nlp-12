import numpy as np
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.steps_per_epoch = len(self.data_loader)
        self.batch_size = self.data_loader.batch_size
        
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('train/loss', *['train/' + m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('val/loss', *['val/' + m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        for step, data in enumerate(tqdm(self.data_loader, desc=f'TRAINING - [{epoch}] EPOCH')):
            input_ids, token_type_ids, attention_mask, targets = data
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            outputs = self.model(inputs)
            
            logits = outputs[0]
            _, preds = torch.max(logits, dim=1)
            
            loss = self.criterion(logits, targets)
            loss.backward()
            
            # https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
            # avoding exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            preds = preds.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            self.train_metrics.update('train/loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update('train/' + met.__name__, met(preds, targets))

            if step % self.log_step == 0:
                self.logger.debug('Train Loss: {:.6f}'.format(
                    loss.item()))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.valid_data_loader, desc=f'VALIDATION - [{epoch}] EPOCH')):
                input_ids, token_type_ids, attention_mask, targets = data
            
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                targets = targets.to(self.device)

                inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
                }
                outputs = self.model(inputs)
                
                logits = outputs[0]
                _, preds = torch.max(logits, dim=1)
                loss = self.criterion(logits, targets)
                
                preds = preds.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

                self.valid_metrics.update('val/loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update('val/' + met.__name__, met(preds, targets))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.steps_per_epoch
        return base.format(current, total, 100.0 * current / total)
