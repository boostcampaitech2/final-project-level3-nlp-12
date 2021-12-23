from os import sep
import pandas as pd
from datasets import load_dataset

DATA_PATH = 'AI-it/korean-hate-speech'
DATA_FILES = {
    "train_comments": "train_hate.csv",
    "train_titles": "train_news_title.txt"
}

def retrieve_comments(keyword: str) -> list:
    result = []
    df_comments = pd.read_csv('data/unlabeled/unlabeled_comments.txt', header=None, encoding='utf-8')
    for comment in df_comments[0]:
        if type(comment) != str:
            continue
        elif keyword in comment:
            result.append(comment)
        elif keyword[1:] in comment:
            result.append(comment)

    return result

if __name__ == '__main__':
    # lst = retrieve_comments('전현무')
    retrieve_comments('전현무')
    # print(lst)