"""Tune Model.
- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
from pathlib import Path
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from optuna.pruners import HyperbandPruner
from subprocess import _args_from_interpreter_flags
import argparse
from transformers import AutoTokenizer
from proj_dataloader import KhsDataLoader
from src.utils.common import write_yaml


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=10, high=10, step=10)
    batch_size = trial.suggest_int("batch_size", low=16, high=32, step=16)
    max_length = trial.suggest_int("max_length", low=64, high=128, step=32)

    return {
        "EPOCHS": epochs,
        "BATCH_SIZE": batch_size,
        "MAX_LENGTH": max_length,
    }


def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_stride = 0
    # MAX_NUM_STRIDE = 5
    # UPPER_STRIDE = 2  # 5(224 example): 224, 112, 56, 28, 14, 7

    # Module 1
    # pretrained model
    # m1 = trial.suggest_categorical("m1", ["Bert", "Electra"])
    m1 = "Electra"
    m1_args = []
    if m1 == "Bert":
        m1_name = "klue/bert-base"
    elif m1 == "Electra":
        m1_name = "beomi/beep-KcELECTRA-base-hate"
    m1_args = [m1_name]
    m1_repeat = 1
    model.append([m1_repeat, m1, m1_args])

    # Module 2
    # Lstm classifier
    m2 = "Lstm"
    m2_args = []
    m2_repeat = 1
    m2_name = "rnn"
    m2_xdim = 768
    m2_hdim = trial.suggest_categorical("m2/hdim", [256, 512, 768, 1024])
    m2_ydim = 3
    m2_n_layer = trial.suggest_int("m2/n_layer", low=1, high=5, step=1)
    m2_dropout = trial.suggest_float("m2/dropout", low=0.1, high=0.5, step=0.1)
    # lstm args : [name, xdim, hdim, ydim, n_layer, dropout]
    m2_args = [m2_name, m2_xdim, m2_hdim, m2_ydim, m2_n_layer, m2_dropout]
    model.append([m2_repeat, m2, m2_args])

    module_info = {}
    module_info["m1"] = {"type": m1, "repeat": m1_repeat}
    module_info["m2"] = {"type": m2, "repeat": m2_repeat}

    return model, module_info


def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    PATH = os.path.join(SAVE_PATH, f"trial{trial.number}")
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    model_config: Dict[str, Any] = {}

    model_config["backbone"], module_info = search_model(trial)
    hyperparams = search_hyperparam(trial)

    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    # Setting data_config to use hyperparameter search and huggingface's datasets library
    data_config: Dict[str, Any] = {}
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["DATA_DIR"] = "AI-it/korean-hate-speech"
    data_config["DATA_FILES"] = {"train": "train_hate.csv", "valid": "dev_hate.csv"}
    data_config["MAX_LENGTH"] = hyperparams["MAX_LENGTH"]
    mean_time = check_runtime(
        model.model,
        hyperparams["MAX_LENGTH"],
        device,
    )

    # Setting Tokenizer
    tokenizer = None
    if module_info["m1"]["type"] == "Bert":
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    elif module_info["m1"]["type"] == "Electra":
        tokenizer = AutoTokenizer.from_pretrained("beomi/beep-KcELECTRA-base-hate")

    # build train and valid dataloader
    dataloader = KhsDataLoader(tokenizer, max_length=data_config["MAX_LENGTH"])
    train_loader = dataloader.get_dataloader(
        name="train",
        data_dir=data_config["DATA_DIR"],
        # data_dir="AI-it/pseudo-labeled-khs",
        data_files=data_config["DATA_FILES"],
        # data_files={"train": "sample_pseudo_labeled_v1.csv"},
        batch_size=data_config["BATCH_SIZE"],
    )
    val_loader = dataloader.get_dataloader(
        name="valid",
        data_dir=data_config["DATA_DIR"],
        data_files=data_config["DATA_FILES"],
        batch_size=data_config["BATCH_SIZE"],
    )

    model_info(model, verbose=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )

    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path=PATH,
        model_config=model_config,
        mean_time=mean_time,
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)
    write_yaml(data_config, "data_config", path=PATH)
    write_yaml(hyperparams, "hyperparams", path=PATH)

    model_info(model, verbose=True)
    return f1_score, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, storage: str = None, save_path=""):
    global SAVE_PATH
    SAVE_PATH = save_path
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    print(device)
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="automl101",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, device), n_trials=500)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--storage", default="", type=str, help="Optuna database storage path."
    )
    parser.add_argument(
        "--path", default="save", type=str, help="model files save path."
    )
    args = parser.parse_args()
    tune(
        args.gpu,
        storage=args.storage if args.storage != "" else None,
        save_path=args.path,
    )
