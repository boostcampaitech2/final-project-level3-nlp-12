import os
import torch
import shutil
from abc import abstractmethod
from numpy import inf
from utils import write_json


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_steps = cfg_trainer["save"]["steps"]
        self.save_limits = cfg_trainer["save"]["limits"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.not_improved_count = 0

        self.checkpoint_dir = cfg_trainer["save"]["dir"]

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def train(self):
        """
        Full training logic.
        """

        raise NotImplementedError

    @abstractmethod
    def _validation(self, step):
        """
        Full validation logic

        :param step: Current step number
        """
        
        raise NotImplementedError

    def _evaluate_performance(self, log):
        # evaluate model performance according to configured metric, save best checkpoint as model_best
        is_best = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (
                    self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                ) or (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. "
                    "Model performance monitoring is disabled.".format(self.mnt_metric)
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                is_best = True
            else:
                self.not_improved_count += 1

        return is_best

    def _save_checkpoint(self, log, is_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'best_model.pt'
        """
        save_path = f'{self.checkpoint_dir}models/{self.config["name"]}/'
        chk_pt_path = save_path + f"steps_{log['steps']}/"
        
        # make path if there isn't
        if not os.path.exists(chk_pt_path):
            os.makedirs(chk_pt_path)
        # delete the oldest checkpoint not to exceed save limits
        if len(os.listdir(save_path)) > self.save_limits:
            shutil.rmtree(os.path.join(
                    save_path,
                    sorted(os.listdir(save_path),key = lambda x : (len(x), x))[0]
                )
            )
        
        self.logger.info("Saving checkpoint: {} ...".format(chk_pt_path))    
        torch.save(self.model, os.path.join(chk_pt_path, "model.pt"))
        torch.save(
            self.optimizer.state_dict(), os.path.join(chk_pt_path, "optimizer.pt")
        )

        # save updated config file to the checkpoint dir
        write_json(self.config._config, os.path.join(chk_pt_path, "config.json"))
        write_json(log, os.path.join(chk_pt_path, "log.json"))

        # save best model.
        if is_best:
            best_path = f'{self.checkpoint_dir}best/{self.config["name"]}/'

            # make path if there isn't
            if not os.path.exists(best_path):
                os.makedirs(best_path)
            # delete old best files
            for file_ in os.listdir(best_path):
                os.remove(best_path + file_)

            self.logger.info("Saving current best: model_best.pt ...")
            torch.save(self.model, os.path.join(best_path, "best_model.pt"))
            torch.save(
                self.optimizer.state_dict(), os.path.join(best_path, "optimizer.pt")
            )

            # save updated config file to the checkpoint dir
            write_json(self.config._config, os.path.join(best_path, "config.json"))
            write_json(log, os.path.join(best_path, "log.json"))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
