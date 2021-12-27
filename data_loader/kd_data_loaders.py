import random
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from torchsampler import ImbalancedDatasetSampler
from transformers import PreTrainedTokenizer
from utils import Preprocess, preprocess
from datasets import load_dataset
from base import BaseDataLoader

LABEL_2_IDX = {
    "none": 0,
    "offensive": 1,
    "hate": 2
}
IDX_2_LABEL = {
    0: "none",
    1: "offensive",
    2: "hate"
}


class KhsDataLoader(DataLoader):
    def __init__(
        self,
        student_tokenizer: PreTrainedTokenizer,
        teacher_tokenizer: PreTrainedTokenizer,
        max_length: int = None
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length if max_length else self.student_tokenizer.model_max_length

    def train_collate_fn(self, input_examples):
        input_texts, input_labels = [], []
        
        for input_example in input_examples:
            text, label = input_example
            input_texts.append(text)
            input_labels.append(label)
        
        st_encoded_texts = self.student_tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask
        
        tc_encoded_texts = self.teacher_tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask

        st_input_ids = st_encoded_texts["input_ids"]
        st_token_type_ids = st_encoded_texts["token_type_ids"]
        st_attention_mask = st_encoded_texts["attention_mask"]
        
        tc_input_ids = tc_encoded_texts["input_ids"]
        tc_token_type_ids = tc_encoded_texts["token_type_ids"]
        tc_attention_mask = tc_encoded_texts["attention_mask"]
        
        return st_input_ids, st_token_type_ids, st_attention_mask, tc_input_ids, tc_token_type_ids, tc_attention_mask, torch.tensor(input_labels)
    
    def valid_collate_fn(self, input_examples):
        input_texts, input_labels = [], []
        
        for input_example in input_examples:
            text, label = input_example
            input_texts.append(text)
            input_labels.append(label)
        
        encoded_texts = self.student_tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
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
            padding="max_length",
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

        sampler = None
        
        if name == "test":
            collate_fn = self.test_collate_fn  
        elif name == "valid":
            collate_fn = self.valid_collate_fn 
        else:
            collate_fn = self.train_collate_fn
            sampler = ImbalancedDatasetSampler(dataset)
            
        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
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


# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id): 
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)