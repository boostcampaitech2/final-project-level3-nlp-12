import wandb
import warnings
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import model.loss as module_loss

class KnowDistTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, student_model, teacher_model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(student_model, criterion, metric_ftns, optimizer, config)
        
        # self.model = student model => assigned in BaseTrainer class.
        self.teacher_model = teacher_model
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.steps_per_epoch = len(self.data_loader)
        self.batch_size = self.data_loader.batch_size
        
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.valid_criterion = getattr(module_loss, 'softmax')

        self.train_metrics = MetricTracker('train/loss', *['train/' + m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('val/loss', *['val/' + m.__name__ for m in self.metric_ftns])

    def train(self):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        step = 0
        self.model.train()
        self.teacher_model.eval()
        
        for epoch, _ in enumerate(range(self.epochs), start = 1):
            for _, data in enumerate(tqdm(self.data_loader, desc=f'TRAINING - [{epoch}] EPOCH')):
                step += 1
                
                st_input_ids, st_token_type_ids, st_attention_mask, tc_input_ids, tc_token_type_ids, tc_attention_mask, targets = data
                
                st_input_ids = st_input_ids.to(self.device)
                st_attention_mask = st_attention_mask.to(self.device)
                st_token_type_ids = st_token_type_ids.to(self.device)
                
                tc_input_ids = tc_input_ids.to(self.device)
                tc_attention_mask = tc_attention_mask.to(self.device)
                tc_token_type_ids = tc_token_type_ids.to(self.device)
                
                targets = targets.to(self.device)
                
                st_inputs = {
                    "input_ids": st_input_ids,
                    "attention_mask": st_attention_mask,
                    "token_type_ids": st_token_type_ids
                }
                tc_inputs = {
                    "input_ids": tc_input_ids,
                    "attention_mask": tc_attention_mask,
                    "token_type_ids": tc_token_type_ids
                }
                
                student_outputs = self.model(st_inputs)
                teacher_outputs = self.teacher_model(tc_inputs)
                
                if isinstance(student_outputs, torch.Tensor):
                    student_logits = student_outputs
                else:
                    student_logits = student_outputs[0]
                    
                if isinstance(teacher_outputs, torch.Tensor):
                    teacher_logits = teacher_outputs
                else:
                    teacher_logits = teacher_outputs[0]
                
                loss = self.criterion(student_logits, targets, teacher_logits)
                
                self.optimizer.zero_grad()
                
                loss.backward()
                
                # https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
                # avoding exploding gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                _, preds = torch.max(student_logits, dim=1)
                preds = preds.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                
                self.train_metrics.update('train/loss', loss.item())
                for met in self.metric_ftns:
                    self.train_metrics.update('train/' + met.__name__, met(preds, targets))

                # activate validation and saving when current step meets 'save steps'
                if step % self.save_steps == 0:
                    log = self.train_metrics.result()
                    
                    if self.do_validation:
                        val_log = self._validation(step)
                        log.update(**{k : v for k, v in val_log.items()})
                    
                    log['epoch'] = epoch
                    log['steps'] = step
                    
                    # visualization log
                    wandb.log(log, step=step)
            
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))
                    
                    is_best = self._evaluate_performance(log)
                    
                    # Early Stopping
                    if self.not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. ""Training stops.".format(self.early_stop))
                        return False
                    
                    self._save_checkpoint(log, is_best)
                    
                    # get back to work again!
                    self.model.train()
                    self.train_metrics.reset()

    def _validation(self, step):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            print(f"VALIDATION - [{step}] STEPS ...")
            for _, data in enumerate(self.valid_data_loader):
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
                
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0]
                    
                loss = self.valid_criterion(logits, targets)
                
                _, preds = torch.max(logits, dim=1)
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