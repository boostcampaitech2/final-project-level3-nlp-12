from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from utils import Preprocess, preprocess
from datasets import load_dataset


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
        
        if name == 'test':
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
    def __init__(self, data, data_type='train'):
        self.data_type = data_type
        self.texts = list(data.texts)
        if self.data_type == 'train' or self.data_type == 'valid':
            self.labels = list(data.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        
        if self.data_type == 'train' or self.data_type == 'valid':
            label = self.labels[index]
            converted_label = LABEL_2_IDX[label]
            
            return text, converted_label
        
        return text
    
        
def get_preprocessed_data(dataset, name):
    if name == 'test':
        preprocessed_sents = preprocess(dataset['comments'])
        out_dataset = pd.DataFrame(
            {
                'texts': preprocessed_sents,
            }
        )
    else:
        preprocessed_sents = preprocess(dataset['comments'])
        out_dataset = pd.DataFrame(
            {
                'texts': preprocessed_sents,
                'labels': dataset['label']
            }
        )
    
    return out_dataset