from fastapi import FastAPI
import uvicorn
from predict import get_pipeline, inference
from database import run_db, insert2db
from load_data import load_data, retrieve_comments
from utils import make_samples


app = FastAPI()
pipe = get_pipeline()
dataset = load_data()

evidence = run_db()

@app.get('/')
def hello_world():
    return {'hello': 'world'}


@app.get('/get_sample/{keyword}')
def get_sample(keyword):
    data = retrieve_comments(keyword, dataset)
    results = inference(data[:10], pipe)
    if results:
        insert2db(keyword, results, evidence)

    return make_samples(results)


if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)