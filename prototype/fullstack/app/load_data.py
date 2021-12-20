import re
import torch
import emoji
from soynlp.normalizer import repeat_normalize

import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore


def load_data():
    dataset = pd.read_csv('/Users/yangjaeug/Desktop/GitHub/Product-Serving/practice/03-streamlit-fastapi/app/data/service/test_data_ver2.csv', low_memory=False)
    dataset['comment'] = preprocess(dataset['comment'])
    dataset = dataset.to_dict('records')
    return dataset

def retrieve_comments(keyword: str, dataset) -> list:
    result = []
    for data in dataset:
        if type(data['comment']) != str:
            continue
        if len(keyword) == 3:
            '''3글자 이름이 들어왔을경우, e.g. 손흥민, 흥민 둘 다 검사'''
            if keyword in data['comment']:
                result.append(data)
            elif keyword[1:] in data['comment']:
                result.append(data)
        else:
            if keyword in data['comment']:
                result.append(data)

    return result

def preprocess(sents):
    preprocessed_sents = []
    
    emojis = set()
    for k in emoji.UNICODE_EMOJI.keys():
        emojis.update(emoji.UNICODE_EMOJI[k].keys())
        
    punc_bracket_pattern = re.compile(f'[\'\"\[\]\(\)]')
    base_pattern = re.compile(f'[^.,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'(http|ftp|https)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )
    
    for sent in sents:
        sent = punc_bracket_pattern.sub(' ', str(sent))
        sent = base_pattern.sub(' ', sent)
        sent = url_pattern.sub('', sent)
        sent = sent.strip()
        sent = repeat_normalize(sent, num_repeats=2)
        preprocessed_sents.append(sent)
            
    return preprocessed_sents


class NhDataloader(DataLoader):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length else self.tokenizer.model_max_length
    
    def collate_fn(self, input_examples):
        input_anc_texts = []
        
        for input_example in input_examples:
            anchor_text = input_example
            input_anc_texts.append(anchor_text)
            
        encoded_texts = self.tokenizer.batch_encode_plus(
            input_anc_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask
        
        return encoded_texts
    
    def get_dataloader(self, data, batch_size, **kwargs):
        dataset = NhDataset(data)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            **kwargs
        )
        
class NhDataset(Dataset):
    def __init__(self, data):
        self.texts = [d['comment'] for d in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]
    
    def get_labels(self):
        return self.labels


# def load_data():
    # DATA_PATH = 'AI-it/khs_service_test'
    # DATA_FILES = {
    #     "data": "test_data_ver2.csv"
    # }
#     dataset = load_dataset(DATA_PATH, data_files=DATA_FILES, use_auth_token=True)
#     return dataset
    




if __name__ == '__main__':
    result = retrieve_comments('전현무')
    
    
