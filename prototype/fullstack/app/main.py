from fastapi import FastAPI
from requests.api import get
from torch.utils.data import DataLoader
import uvicorn
from predict import load_model, inference, load_dataloader
from database import run_db, insert2db
from load_data import load_data, retrieve_comments, NhDataloader
from utils import make_samples


app = FastAPI()
dataset = load_data()
model, classifier = load_model()
evidence = run_db()


@app.get('/')
def hello_world():
    return {'hello': 'world'}


@app.get('/get_sample/{keyword}')
def get_sample(keyword):
    data = retrieve_comments(keyword, dataset)
    inf_dataloader = load_dataloader(data[:128])
    results = inference(model, classifier, inf_dataloader)
    
    res2json = []
    for i, res in enumerate(results[:5]):
        if res == 0:
            continue
        else:
            res2json.append({
                "keyword": keyword,
                'user_id': data[i]['user_id'],
                'comment': data[i]['comment'],
                "label": 'hate',
                'site_name': data[i]['site_name'],
                'site_url': data[i]['site_url'],
                'commented_at': data[i]['commented_at']
            })
  
    if res2json:
        insert2db(keyword, res2json, evidence)
  
    return res2json


if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)  
