import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from proj_utils import Preprocess, preprocess
from datasets import load_dataset


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=default_collate,
    ):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


LABEL_2_IDX = {"none": 0, "offensive": 1, "hate": 2}
IDX_2_LABEL = {0: "none", 1: "offensive", 2: "hate"}


class KhsDataLoader(DataLoader):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length else self.tokenizer.model_max_length

    def train_collate_fn(self, input_examples):
        input_texts, input_labels = [], []
        for input_example in input_examples:
            text, label = input_example
            input_texts.append(text)
            input_labels.append(label)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask

        input_ids = encoded_texts["input_ids"]
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]

        return input_ids, token_type_ids, attention_mask, torch.tensor(input_labels)

    def test_collate_fn(self, input_examples):
        input_texts = []
        for input_example in input_examples:
            text = input_example
            input_texts.append(text)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask

        input_ids = encoded_texts["input_ids"]
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]

        return input_ids, token_type_ids, attention_mask

    def get_dataloader(self, name, data_dir, data_files, batch_size, **kwargs):
        data_files = dict(data_files)
        datasets = load_dataset(data_dir, data_files=data_files, use_auth_token=True)
        dataset = get_preprocessed_data(datasets[name], name)
        dataset = KhsDataset(dataset, name)

        if name == "test":
            collate_fn = self.test_collate_fn
        else:
            collate_fn = self.train_collate_fn

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            **kwargs
        )


class KhsDataLoader_pseudo(DataLoader):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_length: int = None, num_workers=2
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length else self.tokenizer.model_max_length
        self.num_workers = num_workers

    def train_collate_fn(self, input_examples):
        input_texts, input_labels = [], []
        for input_example in input_examples:
            text, label = input_example
            input_texts.append(text)
            input_labels.append(label)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask

        input_ids = encoded_texts["input_ids"]
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]

        return input_ids, token_type_ids, attention_mask, torch.tensor(input_labels)

    def test_collate_fn(self, input_examples):
        input_texts = []
        for input_example in input_examples:
            text = input_example
            input_texts.append(text)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask

        input_ids = encoded_texts["input_ids"]
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]

        return input_ids, token_type_ids, attention_mask

    def get_dataloader(self, name, data_dir, data_files, batch_size, **kwargs):
        data_files = dict(data_files)
        datasets = load_dataset(data_dir, data_files=data_files, use_auth_token=True)
        dataset = get_preprocessed_data(datasets[name], name)
        dataset = KhsDataset_pseudo(dataset, name)

        if name == "test":
            collate_fn = self.test_collate_fn
        else:
            collate_fn = self.train_collate_fn

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            **kwargs
        )


class KhsDataset(Dataset):
    def __init__(self, data, data_type="train"):
        self.data_type = data_type
        self.texts = list(data.texts)
        if self.data_type == "train" or self.data_type == "valid":
            self.labels = list(data.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        if self.data_type == "train" or self.data_type == "valid":
            label = self.labels[index]
            try:
                converted_label = LABEL_2_IDX[label]
            except:
                converted_label = label

            return text, converted_label

        return text


class KhsDataset_pseudo(Dataset):
    def __init__(self, data, data_type="train"):
        self.data_type = data_type
        self.texts = list(data.texts)
        if self.data_type == "train" or self.data_type == "valid":
            self.labels = list(data.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        if self.data_type == "train":
            label = self.labels[index]

            return text, label
        elif self.data_type == "valid":
            label = self.labels[index]
            converted_label = LABEL_2_IDX[label]

            return text, converted_label

        return text


def get_preprocessed_data(dataset, name):
    if name == "test":
        preprocessed_sents = preprocess(dataset["comments"])
        out_dataset = pd.DataFrame(
            {
                "texts": preprocessed_sents,
            }
        )
    else:
        preprocessed_sents = preprocess(dataset["comments"])
        out_dataset = pd.DataFrame(
            {"texts": preprocessed_sents, "labels": dataset["label"]}
        )

    return out_dataset
