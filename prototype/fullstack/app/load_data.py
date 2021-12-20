from pandas import read_csv
from datasets import load_dataset


DATA_PATH = 'AI-it/khs_service_test'
DATA_FILES = {
    "data": "test_data_ver2.csv"
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
    for data in dataset['data']:
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


if __name__ == '__main__':
    result = retrieve_comments('전현무')
    
    
