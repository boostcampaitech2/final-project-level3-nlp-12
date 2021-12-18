from pandas import read_csv
from datasets import load_dataset


DATA_PATH = 'AI-it/korean-hate-speech'
DATA_FILES = {
    "data": "unlabeled_data.csv"
}

def load_data():
    dataset = load_dataset(DATA_PATH, data_files=DATA_FILES, use_auth_token=True)
    return dataset
    

def retrieve_comments(keyword: str, dataset) -> list:
    """Retrieve comments correspond to keyword from unlabeled comments.txt

    Parameters
    ----------
    keyword : name or something you want to find

    Returns
    -------
    list of unlabeled comments contain keyword
    """
    result = []
    for comment in dataset['data']['comment']:
        if type(comment) != str:
            continue
        elif keyword in comment:
            result.append(comment)
        elif keyword[1:] in comment:
            result.append(comment)

    return result


if __name__ == '__main__':
    result = retrieve_comments('전현무')
    
    
